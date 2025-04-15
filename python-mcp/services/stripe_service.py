import stripe
import os  # Add this import
from os import environ as env
from services.logging_config import get_logger
from services.url_utils import clean_url
import asyncio
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from uuid import uuid4

from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

class StripeService:
    _instance = None
    _client = None
    _api_key = None
    _secret_key = None
    
    # Map product IDs to tiers
    PRODUCT_TIER_MAP = {
        'prod_RIPUkyBY4dMZpX': 'pro',
        'prod_RIPVIBFgS4K7jh': 'business_plus'
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StripeService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Stripe configuration"""
        ENV_FILE = find_dotenv()
        if ENV_FILE:
            load_dotenv(ENV_FILE)

        # Initialize both API keys
        self._api_key = clean_url(env.get("STRIPE_API_KEY"))
        logger.debug(f"Initialized STRIPE_API_KEY: {self._api_key}")
        self._secret_key = clean_url(env.get('STRIPE_SECRET_KEY'))  # Secret key
        logger.debug(f"Initialized STRIPE_SECRET_KEY: {self._secret_key}")
        
        if not self._api_key or not self._secret_key:
            logger.warning("Missing Stripe API keys. Both STRIPE_API_KEY and STRIPE_SECRET_KEY are required.")
        
        # Initialize stripe with the secret key
        stripe.api_key = self._secret_key
        self._client = stripe.StripeClient(self._secret_key)
        
        # Log meter configuration
        try:
            # Get all meters configuration
            meters = self._client.billing.meters.list()
            logger.info("Available Stripe meters configuration:")
            for meter in meters:
                logger.debug(f"""
                    Meter ID: {meter.id}
                    Name: {meter.display_name}
                    Customer Mapping Key: {meter.customer_mapping.event_payload_key if hasattr(meter, 'customer_mapping') else 'N/A'}
                    Value Key: {meter.value_settings.event_payload_key if hasattr(meter, 'value_settings') else 'N/A'}
                    Full config: {meter}
                """)
        except Exception as e:
            logger.error(f"Error fetching meter configuration: {str(e)}")

        logger.info("Initialized Stripe service")
        logger.debug(f"Initialized PRODUCT_TIER_MAP: {self.PRODUCT_TIER_MAP}")

    @property
    def client(self):
        """Get the Stripe client instance"""
        return self._client

    async def get_customer_tier(self, stripe_customer_id):
        """
        Get the customer's current subscription tier from Stripe
        Returns: 'pro', 'business_plus', or None for free trial
        """
        try:
            # Run the synchronous Stripe operation in a thread pool
            subscriptions = await asyncio.to_thread(
                self.client.subscriptions.list,
                customer=stripe_customer_id,
                status='active'
            )
            
            if not subscriptions or not subscriptions.data:
                logger.info(f"No active subscriptions found for customer {stripe_customer_id}")
                return None
                
            subscription = subscriptions.data[0]
            items = subscription.items.data[0]
            product_id = items.price.product
            
            logger.info(f"Found product ID: {product_id}")
            
            tier = self.PRODUCT_TIER_MAP.get(product_id)
            logger.info(f"Mapped tier: {tier} for product ID: {product_id}")
            
            return tier
            
        except Exception as e:
            logger.error(f"Error getting customer tier from Stripe: {str(e)}")
            logger.exception("Full traceback:")
            return None

    async def send_meter_event(self, event_name: str, stripe_customer_id: str, value: int = 1):
        """
        Sends a meter event to Stripe using v1 API
        Returns: Response object if successful, None if failed
        """
        if not self._secret_key:
            logger.warning("Missing Stripe secret key. STRIPE_SECRET_KEY is required for meter events.")
            return None
        
        if not stripe_customer_id:
            logger.error("stripe_customer_id is missing.")
            return None

        try:
            # Configure the client with the secret key
            stripe.api_key = self._secret_key
            
            # Log the meter configuration for this specific event
            try:
                meters = await asyncio.to_thread(self._client.billing.meters.list)
                relevant_meter = next((m for m in meters if m.display_name == event_name), None)
                if relevant_meter:
                    logger.info(f"""
                        Using meter configuration:
                        Event name: {event_name}
                        Customer mapping key: {relevant_meter.customer_mapping.event_payload_key if hasattr(relevant_meter, 'customer_mapping') else 'N/A'}
                        Value key: {relevant_meter.value_settings.event_payload_key if hasattr(relevant_meter, 'value_settings') else 'N/A'}
                    """)
            except Exception as e:
                logger.warning(f"Could not fetch meter configuration: {str(e)}")
            
            event_identifier = f"evt_{uuid4().hex}"
            
            event_data = {
                "event_name": event_name,
                "identifier": event_identifier,
                "payload": {
                    "stripe_customer_id": stripe_customer_id,
                    "value": str(value)
                }
            }
            
            logger.info(f"Sending meter event with data: {event_data}")
            
            # Run the synchronous Stripe operation in a thread pool
            response = await asyncio.to_thread(
                stripe.billing.MeterEvent.create,
                **event_data
            )
        
            logger.info(f"Successful meter event response: {response}")
            return response
            
        except stripe.error.AuthenticationError as e:
            logger.warning(f"Stripe authentication error (possibly in test mode): {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error sending Stripe meter event: {str(e)}")
            logger.exception("Full traceback:")
            return None

# Create a singleton instance
stripe_service = StripeService()