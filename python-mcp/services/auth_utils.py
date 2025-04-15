from services.user import User
import logging
from auth0.authentication import GetToken
from os import environ as env
from services.logging_config import get_logger
from typing import Dict, Tuple, Optional, Any
from authlib.integrations.flask_client import OAuth
import httpx

logger = get_logger(__name__)

def get_oauth_client(oauth: OAuth, client_type: str) -> Any:
    """Get OAuth client based on client type.

    Args:
        oauth (OAuth): The OAuth instance
        client_type (str): Type of client ('browser_extension' or 'papr_plugin')

    Returns:
        Any: The OAuth client instance

    Raises:
        ValueError: If client_type is not valid
    """
    if client_type == 'browser_extension':
        client = oauth.create_client("auth0_browser_extension")
    elif client_type == 'papr_plugin':
        client = oauth.create_client("auth0_papr_plugin")
    else:
        raise ValueError("Invalid client type")

    logger.info(f"Retrieved OAuth client for {client_type} with client_id: {client.client_id}")
    return client

def determine_client_type(redirect_uri: Optional[str]) -> str:
    """Determine the client type based on the redirect URI.

    Args:
        redirect_uri (Optional[str]): The redirect URI from the request

    Returns:
        str: The determined client type ('browser_extension' or 'papr_plugin')
    """
    if redirect_uri is None:
        # Handle the None case appropriately
        return "papr_plugin"  # or raise an exception

    if 'chromiumapp.org' in redirect_uri:
        return 'browser_extension'
    elif 'chat.openai.com' in redirect_uri:
        return 'papr_plugin'
    return 'papr_plugin'

class CustomGetToken(GetToken):
    def __init__(self, domain: str) -> None:
        # Pass a placeholder or default client_id to the base class constructor
        default_client_id = "default_client_id"
        super().__init__(domain, default_client_id)

    async def authorization_code(self, client_type: str, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Perform the authorization code flow.

        Args:
            client_type (str): Type of client ('browser_extension' or 'papr_plugin')
            code (str): The authorization code you received.
            redirect_uri (str): The redirect URI that you specified when you initiated the authorization code flow.

        Returns:
            Dict[str, Any]: A dictionary containing the access token and ID token.
        """
        # Choose the right client credentials based on client_type
        client_id, client_secret = self.get_client_credentials(client_type)
        # Log inputs
        logger.info(f"Received client_id: {client_id}")
        logger.info(f"Received client_secret: {client_secret[:5]}...")  # Do not log full client_secret!
        logger.info(f"Received code: {code[:5]}...")  # Do not log full refresh_token!
        logger.info(f"Received redirect_uri: {redirect_uri}")  # Do not log full refresh_token!
        url = f"{self.protocol}://{self.domain}/oauth/token"
        data = {
            'grant_type': 'authorization_code',
            'client_id': client_id,
            'client_secret': client_secret,
            'code': code,
            'redirect_uri': redirect_uri,
        }

        # Log the request being made
        logger.info(f"Making POST request to {url} with payload: {data}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            response.raise_for_status()
            return response.json()

    async def refresh_token(self, client_type: str, refresh_token: str) -> Dict[str, Any]:
        """Perform the refresh token flow.

        Args:
            client_type (str): Type of client ('browser_extension' or 'papr_plugin')
            refresh_token (str): The refresh token you received.

        Returns:
            Dict[str, Any]: A dictionary containing the new access token and ID token.
        """
        # Choose the right client credentials based on client_type
        client_id, client_secret = self.get_client_credentials(client_type)
        # Log inputs
        logger.info(f"Received client_id: {client_id}")
        logger.info(f"Received client_secret: {client_secret[:5]}...")  # Do not log full client_secret!
        logger.info(f"Received refresh_token: {refresh_token[:5]}...")  # Do not log full refresh_token!

        url = f"{self.protocol}://{self.domain}/oauth/token"
        payload = {
            'grant_type': 'refresh_token',
            'client_id': client_id,
            'client_secret': client_secret,
            'refresh_token': refresh_token,
        }
        
        # Log the request being made
        logger.info(f"Making POST request to {url} with payload: {payload}...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=payload)
            response.raise_for_status()
            return response.json()

    def get_client_credentials(self, client_type: str) -> Tuple[str, str]:
        """Get client credentials based on client type.

        Args:
            client_type (str): Type of client ('browser_extension' or 'papr_plugin')

        Returns:
            Tuple[str, str]: A tuple containing (client_id, client_secret)

        Raises:
            ValueError: If client_type is not valid
        """
        if client_type == 'browser_extension':
            return env.get("AUTH0_CLIENT_ID_BROWSER", ""), env.get("AUTH0_CLIENT_SECRET_BROWSER", "")
        elif client_type == 'papr_plugin':
            return env.get("AUTH0_CLIENT_ID_PAPR", ""), env.get("AUTH0_CLIENT_SECRET_PAPR", "")
        else:
            raise ValueError("Invalid client type")

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Retrieve user information using the access token.

        Args:
            access_token (str): The access token of the user.

        Returns:
            Dict[str, Any]: A dictionary containing user information.
        """
        url = f"{self.protocol}://{self.domain}/userinfo"
        headers = {'Authorization': f'Bearer {access_token}'}

        # Log the request being made
        logger.info(f"Making GET request to {url} with access token.")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            user_info = response.json()
            logger.info(f"Received user info: {user_info}...")
            return user_info

    async def client_credentials(self, client_id: str, client_secret: str) -> Dict[str, Any]:
        """Perform the client credentials flow for API clients.

        Args:
            client_id (str): The API client ID
            client_secret (str): The API client secret

        Returns:
            Dict[str, Any]: A dictionary containing the access token
        """
        url = f"{self.protocol}://{self.domain}/oauth/token"
        data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
            'audience': f"{self.protocol}://{self.domain}/api/v2/"
        }

        logger.info(f"Making client credentials request for client_id: {client_id}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            response.raise_for_status()
            return response.json()

async def get_user_from_token(
    auth_header: str, 
    client_type: str
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Asynchronously extracts user information from either Bearer or Session token.

    Args:
        auth_header (str): The Authorization header from the request
        client_type (str): Type of client ('browser_extension' or 'papr_plugin')

    Returns:
        Tuple[str, str, Optional[Dict[str, Any]]]: A tuple containing (user_id, sessionToken, user_info)

    Raises:
        ValueError: If the authorization header is invalid or the token is invalid
        AuthenticationError: If token verification fails
    """
    try:
        logger.info(f"Received auth_header: {auth_header}")
        logger.info(f"Received client_type: {client_type}")

        if not auth_header:
            logger.error("No Authorization header provided")
            raise ValueError("Invalid Authorization header")

        if 'Bearer ' not in auth_header and 'Session ' not in auth_header and 'APIKey ' not in auth_header:
            logger.error("Authorization header does not contain 'Bearer ' or 'Session '")
            raise ValueError("Invalid Authorization header")

        if 'Bearer ' in auth_header:
            token = auth_header.split('Bearer ')[1]
            logger.info(f"Got the access_token: {token[:5]}...")

            user_info = await User.verify_access_token(token, client_type)
            if not user_info:
                logger.error("Invalid access token")
                raise ValueError("Invalid access token")

            user_id = user_info['https://papr.scope.com/objectId']
            sessionToken = user_info['https://papr.scope.com/sessionToken']
            return user_id, sessionToken, user_info

        elif 'Session ' in auth_header:
            # Session token
            sessionToken = auth_header.split('Session ')[1]
            logger.info(f"Got the session_token: {sessionToken[:5]}...")

            parse_user = await User.verify_session_token(sessionToken)
            if not parse_user:
                logger.error("Invalid session token")
                raise ValueError("Invalid session token")

            # Get user_id from ParseUserPointer model
            user_id = parse_user.objectId
            logger.info(f"Retrieved user_id: {user_id}")
            
            return user_id, sessionToken, None

        elif 'APIKey ' in auth_header:
            # API key
            api_key = auth_header.split('APIKey ')[1]
            logger.info(f"Got the api_key: {api_key[:5]}...")

            user_info = await User.verify_api_key(api_key)
            if not user_info:
                logger.error("Invalid API key")
                raise ValueError("Invalid API key")
            # Get user_id from ParseUserPointer model
            user_id = user_info['objectId']
            logger.info(f"Retrieved user_id: {user_id}")
            sessionToken = User.get_user_session_by_tenant(user_id)
            logger.info(f"Retrieved session token: {sessionToken[:5]}...")
            return user_id, sessionToken, None
    except ValueError as e:
        logger.error(f"Validation error in get_user_from_token_async: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_user_from_token_async: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise ValueError(f"Authentication failed: {str(e)}")
