"""
FHIR Client

HTTP client for interacting with HAPI FHIR server.
Includes retry logic, error handling, and batch operations.
"""

import time
from typing import List, Dict, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fhir.resources.observation import Observation
from fhir.resources.riskassessment import RiskAssessment
from fhir.resources.flag import Flag

from config.settings import settings
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class FHIRClientError(Exception):
    """Raised when FHIR client operation fails"""
    pass


class FHIRClient:
    """
    Client for HAPI FHIR server operations.

    Features:
    - Automatic retry with exponential backoff
    - Batch operations
    - Resource validation
    - Error handling

    Example:
        >>> client = FHIRClient()
        >>> obs_id = client.post_observation(observation)
        >>> print(f"Created Observation/{obs_id}")
    """

    def __init__(
        self,
        base_url: str = None,
        timeout: int = None,
        retry_attempts: int = None
    ):
        """
        Initialize FHIR client.

        Args:
            base_url: FHIR server base URL (default: from settings)
            timeout: Request timeout in seconds (default: from settings)
            retry_attempts: Number of retry attempts (default: from settings)
        """
        self.base_url = (base_url or settings.FHIR_SERVER_URL).rstrip('/')
        self.timeout = timeout or settings.FHIR_TIMEOUT
        self.retry_attempts = retry_attempts or settings.FHIR_RETRY_ATTEMPTS

        # Create session with retry logic
        self.session = self._create_session()

        logger.info(f"FHIR client initialized: {self.base_url}")

    def _create_session(self) -> requests.Session:
        """
        Create requests session with retry logic.

        Returns:
            Configured requests Session
        """
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=settings.FHIR_RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def check_server_status(self) -> bool:
        """
        Check if FHIR server is accessible.

        Returns:
            True if server is accessible

        Raises:
            FHIRClientError: If server is not accessible
        """
        try:
            response = self.session.get(
                f"{self.base_url}/metadata",
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"FHIR server is accessible: {self.base_url}")
            return True

        except requests.exceptions.RequestException as e:
            raise FHIRClientError(f"FHIR server not accessible: {e}")

    def post_observation(self, observation: Observation) -> str:
        """
        POST Observation resource to FHIR server.

        Args:
            observation: FHIR Observation resource

        Returns:
            Created resource ID

        Raises:
            FHIRClientError: If POST fails
        """
        return self._post_resource("Observation", observation)

    def post_risk_assessment(self, risk_assessment: RiskAssessment) -> str:
        """
        POST RiskAssessment resource to FHIR server.

        Args:
            risk_assessment: FHIR RiskAssessment resource

        Returns:
            Created resource ID

        Raises:
            FHIRClientError: If POST fails
        """
        return self._post_resource("RiskAssessment", risk_assessment)

    def post_flag(self, flag: Flag) -> str:
        """
        POST Flag resource to FHIR server.

        Args:
            flag: FHIR Flag resource

        Returns:
            Created resource ID

        Raises:
            FHIRClientError: If POST fails
        """
        return self._post_resource("Flag", flag)

    def _post_resource(
        self,
        resource_type: str,
        resource: Union[Observation, RiskAssessment, Flag]
    ) -> str:
        """
        Generic POST method for FHIR resources.

        Args:
            resource_type: FHIR resource type
            resource: FHIR resource object

        Returns:
            Created resource ID

        Raises:
            FHIRClientError: If POST fails
        """
        url = f"{self.base_url}/{resource_type}"

        try:
            # Convert to dict for JSON serialization
            resource_dict = resource.dict()

            # POST request
            response = self.session.post(
                url,
                json=resource_dict,
                headers={"Content-Type": "application/fhir+json"},
                timeout=self.timeout
            )

            response.raise_for_status()

            # Extract ID from response
            response_json = response.json()
            resource_id = response_json.get('id')

            if not resource_id:
                raise FHIRClientError(
                    f"No ID in response from server: {response_json}"
                )

            logger.debug(f"Created {resource_type}/{resource_id}")
            return resource_id

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error posting {resource_type}: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nServer response: {error_detail}"
                except:
                    error_msg += f"\nServer response: {e.response.text}"
            raise FHIRClientError(error_msg)

        except requests.exceptions.RequestException as e:
            raise FHIRClientError(f"Request failed for {resource_type}: {e}")

    def batch_post_observations(
        self,
        observations: List[Observation],
        batch_size: int = None
    ) -> List[str]:
        """
        POST multiple Observations in batches.

        Args:
            observations: List of Observation resources
            batch_size: Batch size (default: from settings)

        Returns:
            List of created resource IDs

        Example:
            >>> ids = client.batch_post_observations(observations)
            >>> print(f"Created {len(ids)} observations")
        """
        batch_size = batch_size or settings.FHIR_BATCH_SIZE

        logger.info(
            f"Batch posting {len(observations)} observations "
            f"(batch_size={batch_size})"
        )

        resource_ids = []

        for i in range(0, len(observations), batch_size):
            batch = observations[i:i + batch_size]

            for obs in batch:
                try:
                    resource_id = self.post_observation(obs)
                    resource_ids.append(resource_id)
                except FHIRClientError as e:
                    logger.error(f"Failed to post observation: {e}")
                    # Continue with next observation

            # Small delay between batches to avoid overwhelming server
            if i + batch_size < len(observations):
                time.sleep(0.1)

        logger.info(
            f"Batch post complete: {len(resource_ids)}/{len(observations)} successful"
        )

        return resource_ids

    def get_resource(
        self,
        resource_type: str,
        resource_id: str
    ) -> Dict:
        """
        GET a specific resource by ID.

        Args:
            resource_type: FHIR resource type (e.g., "Observation")
            resource_id: Resource ID

        Returns:
            Resource as dictionary

        Raises:
            FHIRClientError: If GET fails
        """
        url = f"{self.base_url}/{resource_type}/{resource_id}"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            logger.debug(f"Retrieved {resource_type}/{resource_id}")
            return response.json()

        except requests.exceptions.RequestException as e:
            raise FHIRClientError(
                f"Failed to get {resource_type}/{resource_id}: {e}"
            )

    def search_resources(
        self,
        resource_type: str,
        params: Dict[str, str] = None
    ) -> List[Dict]:
        """
        Search for resources.

        Args:
            resource_type: FHIR resource type
            params: Search parameters (e.g., {"subject": "Patient/123"})

        Returns:
            List of matching resources

        Example:
            >>> results = client.search_resources(
            ...     "Observation",
            ...     {"subject": "Patient/123", "code": "8867-4"}
            ... )
        """
        url = f"{self.base_url}/{resource_type}"

        try:
            response = self.session.get(
                url,
                params=params or {},
                timeout=self.timeout
            )
            response.raise_for_status()

            bundle = response.json()

            # Extract entries from bundle
            if 'entry' in bundle:
                resources = [entry['resource'] for entry in bundle['entry']]
                logger.debug(
                    f"Search returned {len(resources)} {resource_type} resources"
                )
                return resources
            else:
                logger.debug(f"Search returned no {resource_type} resources")
                return []

        except requests.exceptions.RequestException as e:
            raise FHIRClientError(f"Search failed for {resource_type}: {e}")


# Example usage
if __name__ == "__main__":
    from src.fhir.converter import create_observation
    from src.fhir.risk_resources import create_risk_assessment, create_risk_flag
    from src.utils.constants import RISK_LEVEL_RED

    # Initialize client
    client = FHIRClient()

    # Check server status
    print("Checking FHIR server status...")
    try:
        client.check_server_status()
        print("✓ FHIR server is accessible\n")
    except FHIRClientError as e:
        print(f"✗ FHIR server not accessible: {e}")
        print("\nMake sure HAPI FHIR server is running:")
        print("  docker-compose up -d fhir-server")
        exit(1)

    # Create and post sample Observation
    print("=== Creating Sample Observation ===")
    obs = create_observation(
        user_id=999,
        date="2023-01-01",
        metric_name="heart_rate_avg",
        value=75
    )

    print("Posting Observation to FHIR server...")
    try:
        obs_id = client.post_observation(obs)
        print(f"✓ Created Observation/{obs_id}\n")

        # Retrieve it back
        print("Retrieving Observation...")
        retrieved_obs = client.get_resource("Observation", obs_id)
        print(f"✓ Retrieved Observation/{obs_id}")
        print(f"  Value: {retrieved_obs['valueQuantity']['value']} "
              f"{retrieved_obs['valueQuantity']['unit']}\n")

    except FHIRClientError as e:
        print(f"✗ Failed: {e}\n")

    # Create and post sample RiskAssessment
    print("=== Creating Sample RiskAssessment ===")
    risk_assessment = create_risk_assessment(
        user_id=999,
        ml_score=0.72,
        risk_level=RISK_LEVEL_RED,
        observation_ids=[obs_id] if obs_id else None
    )

    print("Posting RiskAssessment to FHIR server...")
    try:
        risk_id = client.post_risk_assessment(risk_assessment)
        print(f"✓ Created RiskAssessment/{risk_id}\n")
    except FHIRClientError as e:
        print(f"✗ Failed: {e}\n")

    # Create and post sample Flag
    print("=== Creating Sample Flag ===")
    flag = create_risk_flag(
        user_id=999,
        risk_level=RISK_LEVEL_RED,
        reason="High cardiovascular wellness risk"
    )

    if flag:
        print("Posting Flag to FHIR server...")
        try:
            flag_id = client.post_flag(flag)
            print(f"✓ Created Flag/{flag_id}\n")
        except FHIRClientError as e:
            print(f"✗ Failed: {e}\n")

    # Search for patient's observations
    print("=== Searching for Patient Observations ===")
    try:
        results = client.search_resources(
            "Observation",
            {"subject": "Patient/999"}
        )
        print(f"✓ Found {len(results)} observations for Patient/999")
    except FHIRClientError as e:
        print(f"✗ Search failed: {e}")
