# user.py
from flask_login import UserMixin
from jose import jwt  # Add this import
import requests  # Add this import
from auth0.authentication import Users  # Add this import
from dotenv import find_dotenv, load_dotenv
from os import environ as env  # Change this line
import json
from services.logging_config import get_logger
from services.url_utils import clean_url
import asyncio
from datetime import datetime   
from services.stripe_service import stripe_service, stripe  # Import both the service and the module
from uuid import uuid4  # Add this import
from threading import Thread  # Add this import
import httpx  # Import at the top of the file in practice
from typing import Dict, Tuple, Optional, Any, List, Union, TypedDict, Literal
from models.parse_server import ParseUserPointer, InteractionLimits, TierLimits
# Create a logger instance for this module
from services.logger_singleton import LoggerSingleton

# Get logger instance
logger = LoggerSingleton.get_logger(__name__)


ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Initialize Parse client
PARSE_SERVER_URL = clean_url(env.get("PUBLIC_SERVER_URL"))
PARSE_APPLICATION_ID = clean_url(env.get("PARSE_APPLICATION_ID"))
PARSE_MASTER_KEY = clean_url(env.get("PARSE_MASTER_KEY"))

# Update HEADERS to only use required fields
HEADERS = {
    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
    "X-Parse-Master-Key": PARSE_MASTER_KEY,  # Use Master Key for admin operations
    "Content-Type": "application/json"
}

TierType = Literal['pro', 'business_plus', 'free_trial']
InteractionType = Literal['mini', 'premium']

# Add these constants at the top of the file with other constants
PRICE_IDS = {
    'pro': {
        'monthly': 'price_1QYkbKLvxLkj9c6vSascP5yn',
        'yearly': 'price_1QYkafLvxLkj9c6vTXtTry9W',
    },
    'businessPlus': {
        'monthly': 'price_1QYkncLvxLkj9c6vZffzw8JS',
        'yearly': 'price_1QYkoKLvxLkj9c6vOPKvQ89Y',
    }
}

from enum import Enum

class StripeSubscriptionStatus(Enum):
    INCOMPLETE = 'incomplete'  # Initial payment failed, awaiting payment
    INCOMPLETE_EXPIRED = 'incomplete_expired'  # Initial payment failed and expired
    TRIALING = 'trialing'  # In trial period
    ACTIVE = 'active'  # Subscription is active
    PAST_DUE = 'past_due'  # Payment failed but retrying
    CANCELED = 'canceled'  # Subscription canceled
    UNPAID = 'unpaid'  # Payment failed and no more attempts
    PAUSED = 'paused'  # Trial ended without payment method


class UserEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, User):
            return obj.to_dict()  # Use the to_dict method to serialize
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

class User(UserMixin):
    def __init__(self, id):
        self.objectId = id
        self.id = id  # Keep this for UserMixin compatibility

    def to_dict(self):
        # Convert to a dictionary representation
        return {'objectId': self.objectId, 'id': self.id}

    @staticmethod
    def get(user_id):
        # Here you should fetch the user from your Parse Server DB using the user_id
        # For now, let's just return a User object
        return User(user_id)

    @staticmethod
    async def get_user_async(user_id: str):
        """
        Asynchronously fetch user information from Parse Server
        Returns user's display name or full name
        """
        url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                user_data = response.json()
                return {
                    'name': user_data.get('displayName') or user_data.get('fullName') or 'Unknown User'
                }
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch user data: {str(e)}")
                raise Exception(f"Failed to fetch user data: {str(e)}")

    @staticmethod
    async def get_company_async(user_id: str, workspace_id: str = None, session_token: str = None):
        """
        Asynchronously fetch company information from Parse Server
        First gets workspace info, then fetches the linked company's display name
        """
        if not workspace_id:
            if not session_token:
                session_token = await User.lookup_user_token(user_id)
            workspace_id = await User.get_selected_workspace_id_async(user_id, session_token)
            if not workspace_id:
                return None

        # First, get the workspace to find the company pointer
        workspace_url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace/{workspace_id}"
        params = {
            "include": "company"  # This will include the company object in the response
        }
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Session-Token": session_token,  
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(workspace_url, headers=headers, params=params)
                response.raise_for_status()
                workspace_data = response.json()
                company_data = workspace_data.get('company')
                if company_data:
                    return company_data.get('displayName', 'Unknown Company')
                return None
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch workspace/company data: {str(e)}")
                raise Exception(f"Failed to fetch workspace/company data: {str(e)}")
        
    @staticmethod
    def get_workspaces_for_user(user_id: str):
        # Prepare the URL to query workspace_follower class
        url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"

        # Prepare the query to filter by user
        query = {
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }

        # Prepare the headers
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        # Send the request
        response = requests.get(url, headers=headers, params={"where": json.dumps(query)})

        # Check the response
        if response.status_code == 200:
            data = response.json()
            workspace_ids = [follower['workspace']['objectId'] for follower in data['results']]
            return workspace_ids
        else:
            raise Exception(f"Failed to fetch workspaces for user: {response.text}")
        
    async def get_roles_async(self) -> List[str]:
        """
        Asynchronously fetch roles from the _User class using httpx
        
        Returns:
            List[str]: List of role names for the user
        """
        url = f"{PARSE_SERVER_URL}/parse/classes/_Role"
        
        params = {
            "where": json.dumps({
                "users": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": self.id
                }
            })
        }
        
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                
                roles_data = response.json()
                return [role['name'] for role in roles_data.get('results', [])]
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred while fetching roles: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching roles: {e}")
            return []

    @staticmethod
    async def get_workspaces_for_user_async(user_id: str) -> List[str]:
        """
        Asynchronously fetch workspaces for a user using httpx
        
        Args:
            user_id (str): The ID of the user to fetch workspaces for
            
        Returns:
            List[str]: List of workspace IDs the user has access to
            
        Raises:
            HTTPError: If the request fails
        """
        url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"

        query = {
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            }
        }

        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        params = {
            "where": json.dumps(query)
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:  
                response = await client.get(
                    url,
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                workspace_ids = [
                    follower['workspace']['objectId'] 
                    for follower in data.get('results', [])
                    if 'workspace' in follower and 'objectId' in follower['workspace']
                ]
                return workspace_ids

        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred while fetching workspaces: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching workspaces for user {user_id}: {e}")
            raise
    @staticmethod
    async def verify_api_key(api_key: str) -> Optional[ParseUserPointer]:
        """
        Asynchronously verify an API key and return the associated user.
        """
        url = f"{PARSE_SERVER_URL}/parse/users"
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,            
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        
        # Query parameters to find user with matching API key
        params = {
            "where": json.dumps({
                "userAPIkey": api_key
            })
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                user_data = response.json()
                
                # Check if any users were found
                if user_data.get('results') and len(user_data['results']) > 0:
                    return user_data['results'][0]  # Return the first matching user
                return None  # Return None if no user found with this API key
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred while verifying API key: {e}")
                raise
            except Exception as e:
                logger.error(f"Error verifying API key: {e}")
                raise
    
    @staticmethod
    async def verify_session_token(session_token: str) -> Optional[ParseUserPointer]:
        """
        Asynchronously verify a session token and return the associated user.
        
        Args:
            session_token (str): The session token to verify
            
        Returns:
            Optional[User]: User instance if token is valid, None otherwise
        """
        logger.info(f"parse server URL: {PARSE_SERVER_URL}")
        logger.info(f"Verifying session token: {session_token[:5]}...")
        logger.info(f"PARSE_APPLICATION_ID: {env.get('PARSE_APPLICATION_ID')}")
        
        url = f"{PARSE_SERVER_URL}/parse/users/me"
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Session-Token": session_token  # Session token is sufficient for authentication
        }
        
        logger.info(f"Making request to {url}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                logger.info(f"Response status code: {response.status_code}")
                logger.debug(f"Response body: {response.text}")
                
                response.raise_for_status()
                
                user_data = response.json()
                logger.debug(f"User data: {user_data}")
                return User(user_data['objectId'])
                
            except httpx.HTTPError as e:
                logger.error(f"Invalid session token. Error: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error verifying session token: {str(e)}")
                return None


    @staticmethod
    async def verify_access_token(token: str, client_type: str) -> Dict[str, Any]:
        """
        Asynchronously verify an access token and retrieve user information from Auth0.
        
        Args:
            token (str): The access token to verify
            client_type (str): The type of client ('browser_extension' or 'papr_plugin')
            
        Returns:
            Dict[str, Any]: User information from Auth0
            
        Raises:
            ValueError: If client_type is invalid
        """
        if client_type == 'browser_extension':
            client_id = clean_url(env.get("AUTH0_CLIENT_ID_BROWSER"))
        elif client_type == 'papr_plugin':
            client_id = clean_url(env.get("AUTH0_CLIENT_ID_PAPR"))
        else:
            raise ValueError("Invalid client type")

        auth0_domain = clean_url(env.get("AUTH0_DOMAIN"))
        url = f"https://{auth0_domain}/userinfo"
        headers = {'Authorization': f'Bearer {token}'}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                user_info = response.json()
                logger.info(f"Received user_info: {user_info}")
                return user_info
        except httpx.HTTPError as e:
            logger.error(f"Failed to verify access token: {str(e)}")
            raise ValueError("Failed to verify access token") from e

    
    @staticmethod
    async def store_auth_token(access_token: str, user_id: str, session_token: str):
        """Store the access token for a user in Parse Server.

        Args:
            access_token (str): The access token to store
            user_id (str): The user's ID
            session_token (str): The session token for authentication

        Raises:
            Exception: If the update fails
        """
        # Prepare the URL
        url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"
        logger.info(f"url for parseServer: {url}")

        # Prepare the headers
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Session-Token": session_token,
            "Content-Type": "application/json"
        }

        # Prepare the data
        data = {
            "access_token": access_token
        }

        # Send the request
        async with httpx.AsyncClient() as client:
            response = await client.put(url, headers=headers, json=data)
            logger.info(f"added data to Parse Server: {str(response)}")

            # Check the response
            if response.status_code != 200:
                raise Exception(f"Failed to update user: {response.text}")

    @staticmethod
    async def lookup_access_token(user_id: str, session_token: str = None):
        """Look up the access token for a user from Parse Server.

        Args:
            user_id (str): The user's ID
            session_token (str, optional): The session token for authentication. Defaults to None.

        Returns:
            str: The access token if found

        Raises:
            Exception: If the retrieval fails
        """
        # Prepare the URL
        url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"

        # Prepare the headers
        if session_token:
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Session-Token": session_token,
                "Content-Type": "application/json"
            }
        else:
            headers = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY,
                "Content-Type": "application/json"
            }

        # Send the request
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            logger.info(f"query to get user_id from Parse Server: {str(response)}")

            # Check the response
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve user: {response.text}")

            # Parse the response
            data = response.json()
            logger.info(f"response from parse parsed: {str(data)}")

            # Return the access token
            return data.get("access_token")

    @staticmethod
    async def lookup_user_token(user_id: str) -> str:
        """
        Asynchronously lookup the most recent session token for a user.
        
        Args:
            user_id (str): The user's ID to lookup
            
        Returns:
            str: The session token if found
            
        Raises:
            Exception: If no session is found or if the request fails
        """
        # Prepare the URL
        url = f"{PARSE_SERVER_URL}/parse/classes/_Session"

        # Prepare the query parameters
        query_params = {
            "where": json.dumps({
                "user": {
                    "__type": "Pointer",
                    "className": "_User",
                    "objectId": user_id
                }
            }),
            "order": "-createdAt",
            "limit": 1
        }

        # Prepare the headers
        headers = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        logger.info(f"lookup_user_token headers: {headers} params: {query_params}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=query_params)
                response.raise_for_status()
                
                logger.info(f"Query to get user session from Parse Server: {response.url}")
                
                # Parse the response
                data = response.json()
                logger.info(f"Response from Parse Server: {data}")

                # Return the session token if available
                if data.get("results") and len(data["results"]) > 0:
                    return data["results"][0].get("sessionToken")
                else:
                    raise Exception("No session found for the given user ID")
                    
            except httpx.HTTPError as e:
                error_msg = f"Failed to retrieve user session: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)

    @staticmethod
    def save_get_memory_request(query: str, user_id: str, context: str, relation_type: str, metadata, result, neoQuery=None, memory_source=None):
        # Prepare the URL
        url = f"{PARSE_SERVER_URL}/parse/classes/GetMemoryRequests"


        # Prepare the headers
        headers = HEADERS

        # Handle None or empty object for context
        if context is None or context == {}:
            context = []  # Set to an empty array

        # Prepare the data
        data = {
            "query": query,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "context": context,
            "relation_type": relation_type,
            "metadata": metadata,
            "result": result,
            "neoQuery": neoQuery,
            "memorySource": memory_source            
        }

        # Send the request
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Check the response
        if response.status_code != 201:  # 201 Created
            raise Exception(f"Failed to save get_memory request: {response.text}")

        # Log the successful addition
        logger.info(f"Successfully added get_memory request to Parse Server: {response.json()}")

        # Return the result
        return response.json()
    
    def get_roles(self):
        # Fetch roles from the _User class
        url = f"{PARSE_SERVER_URL}/parse/classes/_Role?where={{\"users\":{{\"__type\":\"Pointer\",\"className\":\"_User\",\"objectId\":\"{self.id}\"}}}}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            roles_data = response.json()
            return [role['name'] for role in roles_data['results']]
        return []

    @staticmethod
    def get_user_session_by_tenant(subtenant_id: str):
        # Fetch the session token directly using the subtenant_id (which is the user_id)
        session_url = f"{PARSE_SERVER_URL}/parse/classes/_Session?where={{\"user\":{{\"__type\":\"Pointer\",\"className\":\"_User\",\"objectId\":\"{subtenant_id}\"}}}}"
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }
        session_response = requests.get(session_url, headers=HEADERS)
        logger.info(f"session response: {session_response}")

        if session_response.status_code == 200:
            session_data = session_response.json()
            if session_data['results']:
                logger.info(f"session_data: {session_data}")
                # Find session with longest time before expiring
                session = None
                latest_expiry = None
                for result in session_data['results']:
                    expires_at = result.get('expiresAt')
                    if expires_at and isinstance(expires_at, dict):
                        iso_date = expires_at.get('iso')
                        if iso_date:
                            if not latest_expiry or iso_date > latest_expiry:
                                latest_expiry = iso_date
                                session = result
                if not session and session_data['results']:
                    session = session_data['results'][0]  # Fallback if no expiry times found
                logger.info(f"session: {session}")
                session_token = session.get('sessionToken')
                if session_token:
                    return session_token
                else:
                    logger.error("Session token not found in the response")
                    return None, None
        else:
            logger.error(f"Failed to fetch session token: {session_response.text}")
        return None, None

    @staticmethod
    async def get_acl_for_workspace(
        workspace_id: Optional[str] = None, 
        tenant_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        additional_user_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, List[str]]]:
        """
        Asynchronously retrieves and transforms the ACL for a given workspace.

        Args:
            workspace_id (Optional[str]): The objectId of the workspace
            tenant_id (Optional[str]): The tenantId to query the workspace
            user_id (Optional[str]): The userId to include if no specific access is set
            additional_user_ids (Optional[List[str]]): List of additional userIds to grant access

        Returns:
            Optional[Dict[str, List[str]]]: Transformed ACL dictionary containing:
                - user_read_access: List[str]
                - user_write_access: List[str]
                - workspace_read_access: List[str]
                - workspace_write_access: List[str]
                - role_read_access: List[str]
                - role_write_access: List[str]
                Returns None if workspace is not found or on error
        """
        if workspace_id:
            where_clause = {"objectId": workspace_id}
        elif tenant_id:
            where_clause = {"tenantId": tenant_id}
        else:
            return None

        url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace"
        params = {
            "where": json.dumps(where_clause)
        }
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        user_acl = {user_id: {"read": True, "write": True}} if user_id else {}
        logger.info(f"user_acl: {user_acl}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=HEADERS, params=params)
                response.raise_for_status()
                
                workspace_data = response.json()
                if workspace_data['results']:
                    workspace = workspace_data['results'][0]
                    acl = workspace.get('ACL')
                    
                    if acl is not None:
                        return User.transform_acl(acl, workspace_id, user_id, additional_user_ids)
                    else:
                        # Handle the case where ACL is not set (public access)
                        logger.info(f"No ACL found for workspace_id: {workspace_id}. Assuming public access.")
                        return User.transform_acl(
                            {"*": {"read": True, "write": True}}, 
                            workspace_id, 
                            user_id, 
                            additional_user_ids
                        )
                else:
                    logger.warning(f"No workspace found for query: {where_clause}")
                    return None
                    
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch workspace: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching workspace: {str(e)}")
                return None
    
    @staticmethod
    async def get_acl_for_postMessage(
        workspace_id: Optional[str] = None, 
        post_message_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        additional_user_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, List[str]]]:
        """
        Asynchronously retrieves and transforms the ACL for a given PostMessage.

        Args:
            workspace_id (Optional[str]): The objectId of the workspace
            post_message_id (Optional[str]): The objectId of the PostMessage
            user_id (Optional[str]): The userId to include
            additional_user_ids (Optional[List[str]]): List of additional userIds to grant access

        Returns:
            Optional[Dict[str, List[str]]]: Transformed ACL dictionary containing:
                - user_read_access: List[str]
                - user_write_access: List[str]
                - workspace_read_access: List[str]
                - workspace_write_access: List[str]
                - role_read_access: List[str]
                - role_write_access: List[str]
                Returns None if PostMessage is not found or on error
        """
        if not post_message_id:
            return None

        url = f"{PARSE_SERVER_URL}/parse/classes/PostMessage"
        params = {
            "where": json.dumps({"objectId": post_message_id})
        }
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=HEADERS, params=params)
                response.raise_for_status()
                
                post_message_data = response.json()
                if post_message_data['results']:
                    post_message = post_message_data['results'][0]
                    acl = post_message.get('ACL')
                    
                    if acl is not None:
                        return User.transform_acl(acl, workspace_id, user_id, additional_user_ids)
                    else:
                        # Handle the case where ACL is not set (public access)
                        logger.info(f"No ACL found for post_message_id: {post_message_id}. Assuming public access.")
                        return User.transform_acl(
                            {"*": {"read": True, "write": True}}, 
                            workspace_id, 
                            user_id, 
                            additional_user_ids
                        )
                else:
                    logger.warning(f"No PostMessage found for id: {post_message_id}")
                    return None
                    
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch PostMessage: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching PostMessage: {str(e)}")
                return None
    
    @staticmethod
    async def get_acl_for_post(
        workspace_id: Optional[str] = None, 
        post_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        additional_user_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, List[str]]]:
        """
        Asynchronously retrieves and transforms the ACL for a given Post.

        Args:
            workspace_id (Optional[str]): The objectId of the workspace
            post_id (Optional[str]): The objectId of the Post
            user_id (Optional[str]): The userId to include
            additional_user_ids (Optional[List[str]]): List of additional userIds to grant access

        Returns:
            Optional[Dict[str, List[str]]]: Transformed ACL dictionary containing:
                - user_read_access: List[str]
                - user_write_access: List[str]
                - workspace_read_access: List[str]
                - workspace_write_access: List[str]
                - role_read_access: List[str]
                - role_write_access: List[str]
                Returns None if Post is not found or on error
        """
        if not post_id:
            return None

        url = f"{PARSE_SERVER_URL}/parse/classes/Post"
        params = {
            "where": json.dumps({"objectId": post_id})
        }
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=HEADERS, params=params)
                response.raise_for_status()
                
                post_data = response.json()
                if post_data['results']:
                    post = post_data['results'][0]
                    acl = post.get('ACL')
                    
                    if acl is not None:
                        return User.transform_acl(acl, workspace_id, user_id, additional_user_ids)
                    else:
                        # Handle the case where ACL is not set (public access)
                        logger.info(f"No ACL found for post_id: {post_id}. Assuming public access.")
                        return User.transform_acl(
                            {"*": {"read": True, "write": True}}, 
                            workspace_id, 
                            user_id, 
                            additional_user_ids
                        )
                else:
                    logger.warning(f"No Post found for id: {post_id}")
                    return None
                    
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch Post: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching Post: {str(e)}")
                return None

    
    @staticmethod
    def transform_acl(acl: dict, workspace_id: str = None, user_id: str = None, additional_user_ids: list = None):
        """
        Transforms the ACL from the Parse Server into a structured dictionary.

        Parameters:
            acl (dict): The ACL dictionary from Parse Server.
            workspace_id (str): The objectId of the workspace.
            user_id (str, optional): The userId to include if no specific access is set.
            additional_user_ids (list, optional): List of additional userIds to grant access.

        Returns:
            dict: Transformed ACL with access lists.
        """
        user_read_access = []
        user_write_access = []
        workspace_read_access = []
        workspace_write_access = []
        role_read_access = []
        role_write_access = []

        # If ACL is provided, process it normally
        for key, value in acl.items():
            if (key == "*" and workspace_id):
                if value.get("read", True):
                    workspace_read_access.append(workspace_id)
                if value.get("write", True):
                    workspace_write_access.append(workspace_id)
            elif (key.startswith("role:") and workspace_id):
                role_id = key.split(":")[-1]  # Extract the role ID after the last colon
                if value.get("read", True):
                    role_read_access.append(role_id)
                if value.get("write", True):
                    role_write_access.append(role_id)
            else:
                user_id = key
                if value.get("read", True):
                    user_read_access.append(user_id)
                if value.get("write", True):
                    user_write_access.append(user_id)

        # If user_id is provided and no specific user access is set, add it to user access
        if user_id and not user_read_access and not user_write_access:
            user_read_access.append(user_id)
            user_write_access.append(user_id)

        # Add additional_user_ids to both read and write access if provided
        if additional_user_ids:
            user_read_access.extend(additional_user_ids)
            user_write_access.extend(additional_user_ids)

        # Remove duplicates
        user_read_access = list(set(user_read_access))
        user_write_access = list(set(user_write_access))
        workspace_read_access = list(set(workspace_read_access))
        workspace_write_access = list(set(workspace_write_access))
        role_read_access = list(set(role_read_access))
        role_write_access = list(set(role_write_access))

        return {
            "user_read_access": user_read_access,
            "user_write_access": user_write_access,
            "workspace_read_access": workspace_read_access,
            "workspace_write_access": workspace_write_access,
            "role_read_access": role_read_access,
            "role_write_access": role_write_access
        }

    @staticmethod
    async def get_selected_workspace_id_async(user_id: str, session_token: str):
        """Get the workspace ID from the user's selected workspace follower asynchronously"""
        url = f"{PARSE_SERVER_URL}/parse/users/{user_id}"
        
        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Session-Token": session_token,  
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=HEADERS)
                response.raise_for_status()
                user_data = response.json()
                selected_follower = user_data.get('isSelectedWorkspaceFollower')
                
                if not selected_follower:
                    logger.warning(f"No selected workspace follower for user {user_id}")
                    return None

                # Get workspace follower data
                follower_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{selected_follower['objectId']}"
                follower_response = await client.get(follower_url, headers=HEADERS)
                follower_response.raise_for_status()
                follower_data = follower_response.json()
                workspace = follower_data.get('workspace')
                
                if workspace and 'objectId' in workspace:
                    return workspace['objectId']
                
                return None
            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred: {str(e)}")
                return None

    async def check_memory_limits(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """
        Asynchronously checks if the user has exceeded their memory limits based on their subscription.
        
        Returns:
            Optional[Tuple[Dict[str, Any], int]]: Tuple containing response and status code if limit exceeded, 
                                                or None if ok
        """
        try:
            # Get selected workspace follower
            selected_follower_id = await self.get_selected_workspace_follower()
            if not selected_follower_id:
                return {"error": "Unable to determine workspace"}, 400

            # Get workspace follower details
            workspace_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{selected_follower_id}"
            params = {
                "include": "workspace,workspace.subscription"  # Include both workspace and its subscription
            }
            HEADERS = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY,
                "Content-Type": "application/json"
            }   
            
            async with httpx.AsyncClient() as client:
                workspace_response = await client.get(workspace_url, headers=HEADERS, params=params)
                if workspace_response.status_code != 200:
                    logger.error(f"Failed to get workspace follower: {workspace_response.text}")
                    return {"error": "Unable to verify workspace"}, 500

                workspace_data = workspace_response.json()
                workspace = workspace_data.get('workspace')
                if not workspace:
                    return {"error": "Unable to determine workspace"}, 400

                subscription = workspace.get('subscription')
                if not subscription:
                    return {
                        "error": "No subscription found for workspace please go to https://app.papr.ai to sign-up for a subscription"
                    }, 400

                stripe_customer_id = subscription.get('stripeCustomerId')
                is_metered_billing_on = subscription.get('isMeteredBillingOn', False)
                current_memories_count = workspace_data.get('memoriesCount', 0)
                logger.info(f"current_memories_count: {current_memories_count}")
                logger.info(f"subscription: {subscription}")

                # If metered billing is on, no need to check limits
                logger.info(f"is_metered_billing_on: {is_metered_billing_on}")
                if is_metered_billing_on:
                    return None

                try:
                    # Get customer info using stripe directly since we're using the client for billing
                    customer = await asyncio.to_thread(
                        stripe.Customer.retrieve,
                        stripe_customer_id
                    )
                    
                    # Use stripe directly instead of the client
                    subscriptions = await asyncio.to_thread(
                        stripe.Subscription.list,
                        customer=stripe_customer_id,
                        status='active',
                        limit=1
                    )
                    
                    if subscriptions.data:
                        stripe_subscription = subscriptions.data[0]
                        is_trial = stripe_subscription.status == 'trialing'
                        
                        # Update subscription status in Parse if needed
                        if is_trial != (subscription.get('status') == 'trial'):
                            update_url = f"{PARSE_SERVER_URL}/parse/classes/Subscription/{subscription['objectId']}"
                            update_data = {"status": "trial" if is_trial else "active"}
                            await client.put(update_url, headers=HEADERS, json=update_data)
                    else:
                        is_trial = False

                except stripe.error.StripeError as e:
                    logger.error(f"Stripe API error: {str(e)}")
                    created_at = subscription.get('createdAt')
                    if created_at:
                        created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                        is_trial = (datetime.now() - created_date).days <= 7
                    else:
                        is_trial = False

                # Define memory limits per tier
                MEMORY_LIMITS = {
                    'pro': 2500,
                    'business_plus': 20000,
                    'enterprise': float('inf'),  # No limit
                    'free_trial': 2500
                }

                # Get customer tier from Stripe
                customer_tier = await asyncio.to_thread(
                    stripe_service.get_customer_tier,
                    stripe_customer_id
                )
                tier_limit = MEMORY_LIMITS.get('free_trial' if is_trial else customer_tier, MEMORY_LIMITS['free_trial'])

                if current_memories_count >= tier_limit:
                    if customer_tier == 'pro':
                        error_message = (
                            f"You've reached the {tier_limit:,} memory limit for your Pro plan. "
                            "To continue adding memories, you can either:\n"
                            "1. Enable metered billing in your current plan, or\n"
                            "2. Upgrade to Business Plus plan for higher limits\n"
                            "Visit https://app.papr.ai to manage your subscription."
                        )
                    else:
                        error_message = (
                            f"You've reached the {tier_limit:,} memory limit for your "
                            f"{customer_tier.replace('_', ' ').title()} plan. "
                            "Please enable metered billing at https://app.papr.ai inside your settings."
                        )
                    
                    return {
                        "error": "Memory limit reached",
                        "message": error_message,
                        "current_count": current_memories_count,
                        "limit": tier_limit,
                        "tier": customer_tier,
                        "is_trial": is_trial
                    }, 403

                return None  # No limits exceeded

        except Exception as e:
            logger.error(f"Error checking memory limits: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return {"error": "Unable to verify memory limits please visit https://app.papr.ai to manage subscription."}, 500

    async def get_selected_workspace_follower(self) -> Optional[str]:
        """
        Get the workspace follower ID from the user's selected workspace follower asynchronously.
        
        Returns:
            Optional[str]: The workspace follower ID if found, None otherwise
        """
        url = f"{PARSE_SERVER_URL}/parse/users/{self.id}"

        # Get session token for the user
        session_token = await User.lookup_user_token(self.id)
        logger.info(f"session_token: {session_token}")

        if not session_token:
            logger.error("Failed to get session token for user")
            return None

        HEADERS = {
            "X-Parse-Application-Id": PARSE_APPLICATION_ID,
            "X-Parse-Master-Key": PARSE_MASTER_KEY,
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=HEADERS)
                response.raise_for_status()
                
                if response.status_code != 200:
                    logger.error(f"Failed to get user data: {response.text}")
                    return None

                user_data = response.json()
                selected_follower = user_data.get('isSelectedWorkspaceFollower')
                
                follower_id = selected_follower.get('objectId') if selected_follower else None
                logger.info(f"Retrieved workspace follower ID: {follower_id}")
                
                return follower_id

            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred while getting workspace follower: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error getting workspace follower: {str(e)}")
                logger.error("Full traceback:", exc_info=True)
                return None
    

    async def check_interaction_limits(
        self, 
        interaction_type: str = 'mini'
    ) -> Optional[Tuple[Dict[str, Any], int, bool]]:
        """
        Checks if the user has exceeded their interaction limits based on their subscription.
        Also ensures an Interaction object exists for the current month and updates counts.
        
        Args:
            interaction_type (str): either 'mini' or 'premium'
            
       Returns:
            Optional[Tuple[Dict[str, Any], int, bool]]: Tuple containing (response, status_code, is_error)
                                                or None if ok

        """
        try:
            # Get workspace follower with included workspace and subscription
            selected_follower_id = await self.get_selected_workspace_follower()
            if not selected_follower_id:
                return {
                    "error": "No workspace found",
                    "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                }, 403, True

            # Get workspace follower details with included pointers
            workspace_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower/{selected_follower_id}"
            params = {
                "include": "workspace,workspace.subscription,workspace.company"
            }
            HEADERS = {
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY,
                "Content-Type": "application/json"
            }

            logger.info(f"workspace_url: {workspace_url}")

            async with httpx.AsyncClient() as client:
                workspace_response = await client.get(workspace_url, headers=HEADERS, params=params)
                if workspace_response.status_code != 200:
                    logger.error(f"Failed to get workspace follower: {workspace_response.text}")
                    return {
                        "error": "No workspace access",
                        "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                    }, 403, True

                workspace_data = workspace_response.json()
                workspace = workspace_data.get('workspace')
                logger.info(f"workspace: {workspace}")
                if not workspace:
                    return {
                        "error": "No workspace found",
                        "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                    }, 403, True

                subscription = workspace.get('subscription')
                logger.info(f"subscription: {subscription}")
                if not subscription:
                    return {
                        "error": "No active subscription",
                        "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
                    }, 403, True

                stripe_customer_id = subscription.get('stripeCustomerId')
                logger.info(f"check interaction limits stripe_customer_id: {stripe_customer_id}")
                is_metered_billing_on = subscription.get('isMeteredBillingOn', False)
                logger.info(f"check interaction limits is_metered_billing_on: {is_metered_billing_on}")

                # Check and update subscription trial status if needed
                try:
                    # First check all subscriptions (including canceled ones)
                    subscriptions = await asyncio.to_thread(
                        stripe.Subscription.list,
                        customer=stripe_customer_id,
                        status='all',  # Get all subscriptions including canceled
                        limit=1,
                        expand=['data.latest_invoice']  # Get invoice info to check payment status
                    )
                    logger.info(f"check interaction limits subscriptions: {subscriptions}")

                    if not subscriptions.data:
                        logger.info("No subscription found - creating new trial subscription")
                        # Create a new subscription with trial
                        try:
                            # Use the monthly pro price ID from our constants
                            price_id = PRICE_IDS['pro']['monthly']
                            logger.info(f"Creating trial subscription with price_id: {price_id}")
                            
                            # Create the subscription with trial settings
                            new_subscription = await asyncio.to_thread(
                                stripe.Subscription.create,
                                customer=stripe_customer_id,
                                items=[{'price': price_id}],
                                trial_period_days=21,
                                trial_settings={
                                    'end_behavior': {
                                        'missing_payment_method': 'cancel'
                                    }
                                },
                                payment_settings={
                                    'payment_method_types': ['card'],
                                    'save_default_payment_method': 'on_subscription'
                                },
                                collection_method='charge_automatically'
                            )
                            logger.info(f"Created new trial subscription: {new_subscription.id}")
                            
                            # Update Parse subscription record
                            update_url = f"{PARSE_SERVER_URL}/parse/classes/Subscription/{subscription['objectId']}"
                            update_data = {
                                "status": "trial",
                                "tier": "pro",
                                "trialEndsAt": datetime.fromtimestamp(new_subscription.trial_end).isoformat()
                            }
                            await client.put(update_url, headers=HEADERS, json=update_data)
                            
                            # Store welcome message but continue execution
                            welcome_message = {
                                "message": "Welcome to Papr! You've been enrolled in a 21-day Pro trial. Visit https://app.papr.ai to add your payment method and continue using Papr after the trial.",
                                "trial_started": True,
                                "days_remaining": 21,
                                "trial_end": new_subscription.trial_end
                            }
                            
                            # Continue with rest of the code for metered events...
                            
                        except stripe.error.StripeError as e:
                            logger.error(f"Failed to create trial subscription: {str(e)}")
                            return {
                                "error": "Subscription setup failed",
                                "message": "Failed to set up your trial subscription. Please visit https://app.papr.ai to try again."
                            }, 403, True
                    
                    if subscriptions.data:
                        existing_subscription = subscriptions.data[0]
                        status = StripeSubscriptionStatus(existing_subscription.status)
                        logger.info(f"check memory limits status: {status}")
                        # Case 1: User has active or past_due subscription
                        if status in [StripeSubscriptionStatus.ACTIVE, StripeSubscriptionStatus.TRIALING]:
                            logger.info(f"check memory limits active or past due")
                            is_trial = False
                                       
                        # Case 3: Subscription needs attention
                        elif status in [
                            StripeSubscriptionStatus.CANCELED,
                            StripeSubscriptionStatus.INCOMPLETE,
                            StripeSubscriptionStatus.INCOMPLETE_EXPIRED,
                            StripeSubscriptionStatus.UNPAID,
                            StripeSubscriptionStatus.PAUSED,
                            StripeSubscriptionStatus.PAST_DUE
                        ]:
                            status_messages = {
                                StripeSubscriptionStatus.CANCELED: "Your subscription has been canceled",
                                StripeSubscriptionStatus.INCOMPLETE: "Your subscription setup was not completed",
                                StripeSubscriptionStatus.INCOMPLETE_EXPIRED: "Your initial subscription setup was not completed",
                                StripeSubscriptionStatus.UNPAID: "Your subscription has unpaid invoices",
                                StripeSubscriptionStatus.PAUSED: "Your subscription is paused",
                                StripeSubscriptionStatus.PAST_DUE: "Your subscription is past due"
                            }
                            return {
                                "error": "Subscription required",
                                "message": f"{status_messages[status]}. Please visit https://app.papr.ai to reactivate your subscription and continue using Papr.",
                                "subscription_status": status.value
                            }, 403, True
                        
                        # Case 4: New user needs trial setup
                        else:
                            logger.info(f"check memory limits new user needs trial setup")
                            # Update subscription with trial settings
                            updated_subscription = await asyncio.to_thread(
                                stripe.Subscription.modify,
                                existing_subscription.id,
                                trial_period_days=21,
                                trial_settings={
                                    'end_behavior': {
                                        'missing_payment_method': 'cancel'
                                    }
                                },
                                payment_settings={
                                    'payment_method_types': ['card'],
                                    'save_default_payment_method': 'on_subscription'
                                },
                                collection_method='charge_automatically'
                            )
                            logger.info(f"check memory limits updated_subscription: {updated_subscription}")
                            
                            # Update Parse subscription record
                            update_url = f"{PARSE_SERVER_URL}/parse/classes/Subscription/{subscription['objectId']}"
                            update_data = {
                                "status": "trial",
                                "tier": "pro",
                                "trialEndsAt": datetime.fromtimestamp(updated_subscription.trial_end).isoformat()
                            }
                            await client.put(update_url, headers=HEADERS, json=update_data)
                            
                            # Store welcome message but don't return yet
                            welcome_message = {
                                "message": "Welcome to Papr! You've been enrolled in a 21-day Pro trial. Visit https://app.papr.ai to add your payment method and continue using Papr after the trial.",
                                "trial_started": True,
                                "days_remaining": 21,
                                "trial_end": new_subscription.trial_end
                            }
                            # Continue execution...

                except stripe.error.StripeError as e:
                    logger.error(f"Stripe API error: {str(e)}")
                    created_at = subscription.get('createdAt')
                    if created_at:
                        created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                        is_trial = (datetime.now() - created_date).days <= 7
                    else:
                        is_trial = False

                # Get current month's usage and ensure Interaction object exists
                current_date = datetime.now()
                current_month = current_date.month
                current_year = current_date.year

                # Query current month's interaction
                interaction_url = f"{PARSE_SERVER_URL}/parse/classes/Interaction"
                query = {
                    "where": json.dumps({
                        "user": {
                            "__type": "Pointer",
                            "className": "_User",
                            "objectId": self.id
                        },
                        "workspace": {
                            "__type": "Pointer",
                            "className": "WorkSpace",
                            "objectId": workspace.get('objectId')
                        },
                        "type": interaction_type,
                        "month": current_month,
                        "year": current_year
                    })
                }
                
                interaction_response = await client.get(interaction_url, headers=HEADERS, params=query)
                if interaction_response.status_code != 200:
                    logger.error(f"Failed to get interaction count: {interaction_response.text}")
                    return {"error": "Unable to verify usage limits"}, 500, True

                interactions = interaction_response.json().get('results', [])
                logger.info(f"interactions: {interactions}")
                logger.info(f"mini interaction count: {interactions[0].get('count', 0) if interactions else 0}")
                
                if not interactions:
                    # Create new interaction record if none exists
                    new_interaction = {
                        "workspace": {
                            "__type": "Pointer",
                            "className": "WorkSpace",
                            "objectId": workspace.get('objectId')
                        },
                        "user": {
                            "__type": "Pointer",
                            "className": "_User",
                            "objectId": self.id
                        },
                        "type": interaction_type,
                        "month": current_month,
                        "year": current_year,
                        "count": 1
                    }
                    # Only add company pointer if it exists in workspace
                    if workspace.get('company'):
                        new_interaction["company"] = {
                            "__type": "Pointer",
                            "className": "Company",
                            "objectId": workspace['company']['objectId']
                        }
                    if subscription.get('objectId'):
                        new_interaction["subscription"] = {
                            "__type": "Pointer",
                            "className": "Subscription",
                            "objectId": subscription.get('objectId')
                        }
                    
                    create_response = await client.post(interaction_url, headers=HEADERS, json=new_interaction)
                    if create_response.status_code != 201:
                        logger.error(f"Failed to create interaction record: {create_response.text}")
                        return {"error": "Unable to create usage record"}, 500, True
                    current_count = 1
                else:
                    # Increment the existing interaction count
                    interaction = interactions[0]
                    current_count = interaction.get('count', 0) + 1
                    
                    # Update the interaction record
                    update_url = f"{interaction_url}/{interaction['objectId']}"
                    update_data = {"count": current_count}
                    update_response = await client.put(update_url, headers=HEADERS, json=update_data)
                    
                    if update_response.status_code != 200:
                        logger.error(f"Failed to update interaction count: {update_response.text}")
                        return {"error": "Unable to update usage record"}, 500, True

                # If metered billing is on, update count and send meter event
                if is_metered_billing_on:
                    # Run meter event in background task
                    async def send_meter_event():
                        try:
                            meter_response = await stripe_service.send_meter_event(
                                event_name=f"papr_{interaction_type}_interactions",
                                value=1,
                                stripe_customer_id=stripe_customer_id,
                            )
                            if meter_response is None:
                                logger.warning("Failed to send meter event to Stripe, but continuing")
                        except Exception as e:
                            logger.error(f"Error sending meter event: {str(e)}")

                    asyncio.create_task(send_meter_event())
                    # At the very end of the method
                    if 'welcome_message' in locals():
                        return welcome_message, 200, False
                    return None, 200, False  # No limits exceeded

                # Get customer tier from Stripe
                customer_tier = await stripe_service.get_customer_tier(stripe_customer_id)

                TIER_LIMITS: TierLimits = {
                    'pro': {'mini': 2500, 'premium': 500},
                    'business_plus': {'mini': 5000, 'premium': 1000},
                    'free_trial': {'mini': 2500, 'premium': 500}
                }

                # Check subscription status from Stripe
                try:
                    customer = await asyncio.to_thread(
                        stripe.Customer.retrieve,
                        stripe_customer_id
                    )
                    subscriptions = await asyncio.to_thread(
                        stripe.Subscription.list,
                        customer=stripe_customer_id,
                        limit=1
                    )
                    
                    if subscriptions.data:
                        stripe_subscription = subscriptions.data[0]
                        is_trial = stripe_subscription.status == 'trialing'
                        
                        # Update subscription status in Parse if needed
                        if is_trial != (subscription.get('status') == 'trial'):
                            update_url = f"{PARSE_SERVER_URL}/parse/classes/Subscription/{subscription['objectId']}"
                            update_data = {
                                "status": "trial" if is_trial else "active",
                                "tier": customer_tier
                            }
                            await client.put(update_url, headers=HEADERS, json=update_data)
                    else:
                        is_trial = False
                except stripe.error.StripeError as e:
                    logger.error(f"Stripe API error: {str(e)}")
                    created_at = subscription.get('createdAt')
                    if created_at:
                        created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                        is_trial = (datetime.now() - created_date).days <= 7
                    else:
                        is_trial = False

                # In the method:
                tier_limits: InteractionLimits = TIER_LIMITS.get(
                    'free_trial' if is_trial else customer_tier, 
                    TIER_LIMITS['free_trial']
                )

                if current_count >= tier_limits[interaction_type]:
                    if customer_tier == 'pro':
                        error_message = (
                            f"You've reached the {tier_limits[interaction_type]:,} {interaction_type} interactions limit. "
                            "To continue, you can either:\n"
                            "1. Enable metered billing in your current plan, or\n"
                            "2. Upgrade to Business Plus plan for higher limits\n"
                            "Visit https://app.papr.ai to manage your subscription in your settings."
                        )
                    else:
                        error_message = (
                            f"You've reached the {tier_limits[interaction_type]:,} {interaction_type} interactions limit for your "
                            f"{customer_tier.replace('_', ' ').title()} plan this month. "
                            "Please enable metered billing at https://app.papr.ai to continue."
                        )
                    
                    return {
                        "error": "Interaction limit reached, please go to https://app.papr.ai to upgrade to a pro or team plan to be able to use Papr.",
                        "message": error_message,
                        "current_count": current_count,
                        "limit": tier_limits[interaction_type],
                        "tier": customer_tier,
                        "is_trial": is_trial
                    }, 403, True

                 # At the end of the method, after metered events
                if 'welcome_message' in locals():
                    return welcome_message, 200, False
                return None, 200, False  # No limits exceeded and not an error

        except Exception as e:
            logger.error(f"Error checking interaction limits: {str(e)}")
            return {
                "error": "Subscription required",
                "message": "Please visit https://app.papr.ai to start your free trial and begin using Papr."
            }, 403, True