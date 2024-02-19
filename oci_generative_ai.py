from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator

from langchain_community.llms.utils import enforce_stop_tokens

CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"
VALID_PROVIDERS = ("cohere", "meta")


class OCIAuthType(Enum):
    """OCI authentication types as enumerator."""

    API_KEY = 1
    SECURITY_TOKEN = 2
    INSTANCE_PRINCIPAL = 3
    RESOURCE_PRINCIPAL = 4


class OCIGenAIBase(BaseModel, ABC):
    """Base class for OCI GenAI models"""

    client: Any  #: :meta private:

    auth_type: Optional[str] = "API_KEY"
    """Authentication type, could be 
    
    API_KEY, 
    SECURITY_TOKEN, 
    INSTANCE_PRINCIPLE, 
    RESOURCE_PRINCIPLE

    If not specified, API_KEY will be used
    """

    auth_profile: Optional[str] = "DEFAULT"
    """The name of the profile in ~/.oci/config
    If not specified , DEFAULT will be used 
    """

    model_id: str = None  # type: ignore[assignment]
    """Id of the model to call, e.g., cohere.command"""

    provider: str = None  # type: ignore[assignment]
    """Provider name of the model. Default to None, 
    will try to be derived from the model_id
    otherwise, requires user input
    """

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model"""

    service_endpoint: str = None  # type: ignore[assignment]
    """service endpoint url"""

    compartment_id: str = None  # type: ignore[assignment]
    """OCID of compartment"""

    is_stream: bool = False
    """Whether to stream back partial progress"""

    llm_stop_sequence_mapping: Mapping[str, str] = {
        "cohere": "stop_sequences",
        "meta": "stop",
    }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that OCI config and python package exists in environment."""

        # Skip creating new client if passed in constructor
        if values["client"] is not None:
            return values

        try:
            import oci

            client_kwargs = {
                "config": {},
                "signer": None,
                "service_endpoint": values["service_endpoint"],
                "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
                "timeout": (10, 240),  # default timeout config for OCI Gen AI service
            }

            if values["auth_type"] == OCIAuthType(1).name:
                key_content = '-----BEGIN RSA PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC+zlB/DB0d1xhYs5TQZIJi+AQefhZ2e8+DB02yLFpSTl3msoluP4B0rB9y200bcm+4hZCo6fREGcQuVpPZg9KDLJ2aO3clRyZ9rPSBQ6Hk1Z8ClABky5WmqHul6IVDtlPnn268qP4fVPQkYMy9W688PbAlFsagAOVTI/dvO50IACEP6Omo3dKbqVd8L7YUMgpc2ZNfTUGQxOcBNTVUYpO4qfnPZk0yWVAP6QysOj/M601Vu50BeOHLyzn7W69yQ76mNgn8lgGiDQ1HKCZ2RjPsKO5ikTymCFI7KfuKkW7D0D3XevGtK+roRz7BqgCY9TQiL2TXks1M9LXP0sWZqN4TAgMBAAECggEAAvm4q6KGubZm/MsgIPfQSLbRaaBRvVtE6ygsAAFND0eFk1lpZzy6RheBDgt5clogf+IlVV0AoCc+tfJHr/asS7tgYc1XmE85vG/8CCAD4EushR/Fucyq8sRh2T59BYdMW6pS0cJY5Cgw/ng7+R/65ZeYrZtjUY9yBA/lheiKFFvVQjLkrwXdLxPZ2AmrWUGoo4+K51jrEYJEL6UVlOZ0qAp3jGC7FG5ZT+xmU65iw7oo1MSrfNc1HQVd+GAK59H9Kr9N10wWTck+hvLNrvka2HMXF4f0CPQckgtSf2WIbwp4bviFlycQ2PjP/J4gSLEjIz0IIhFZJjo/WrdOyk6boQKBgQDltmjJ4PVN0AV43ns2X5NZ0NDQ/Bf+tqzql5jfarsFMw+dbD1TZv1nDNGZMpXKJ9tSLWYRCG6Fq5KlagZ99/pAEGEk4RiZiUSRoKnt9LUXCDyrZVYbtNbCwsJGg0zvGO/lbb9t4FxVA+LxXrTxPWttYplIfOR01UbathduDF7VKwKBgQDUpBu1F+16CgDSHpQWxOZbE0DAqWO/fRQN+BJCYODHglDR59l0vnoIgWkMQNp0HRDX2074ToieMqMKXLoR80zBB+/M6Oi3SH1B/VR7MTZ7L5Y1TqDvnn7UkPrb7LVC58TsV3U8vh49NaKvOFiHYzKmFzcVEKqIqu4V+YDZ5x12uQKBgQCk+UL6sYga8Sns64N7wlpV58UAQfyNnu5CsMtGsSVNj6VIPsMwUcftqWQibRTskYFO3HHwC6kscNMp3yJ4d46PUfQig1KiedX37HX+An/H4c+InIeh8FdImoziDUxEbxljIVWY+Mf2+oqTJJw4oQ61CVnimGeOjNmNG7dp/pJg7wKBgHs7pGceN92bQICFgjNz50Bu9v7w7EMEnQO/ee2IMZzONEKCCG91GEonnZahWzkhwyomCi0zbk+obv7JYHiYtSnFjL8HWB8oXpdI1pYSnk1j9bxtFi3of/53Czs7go0EvwZtZR9P2zyEAgVkNUI5RhglqiGGKB7OWO7ZS0h9fYDZAoGAIAz2S8IIM5QSgNgYkb4AVnwFGiCt6/ZgPi2PbkhwiAg0W4AYhe6M9zLOmhkPRpMI5WdyD1dOAbgaJqrp9kQvMaYiRKk05TXyAwJ8A6xERZ+JXmWRJXy3VfJb4K5zrrZdrIo7yWYIu3T59c/HnddgBrbQWA5Dl5HZLx4IOuG1XF8=\n-----END RSA PRIVATE KEY-----'
                config_with_key_content = {
                        "user": 'ocid1.user.oc1..aaaaaaaaslawrnuafvbftc3ew7k7zhm5w757bxplxqra4kwhnjqpln52hlta',
                        "key_content": key_content,
                        "fingerprint": 'c7:96:fa:50:1f:9a:91:97:94:da:ce:3e:7a:4f:54:81',
                        "tenancy": 'ocid1.tenancy.oc1..aaaaaaaad7nju4v2lioxocqnkeqydccehtrswmbsievlnhie2rrqguu5ruxq',
                        "region": 'us-ashburn-1'
                }
                client_kwargs["config"] = config_with_key_content
                print(client_kwargs["config"])
                client_kwargs.pop("signer", None)
            elif values["auth_type"] == OCIAuthType(2).name:

                def make_security_token_signer(oci_config):  # type: ignore[no-untyped-def]
                    pk = oci.signer.load_private_key_from_file(
                        oci_config.get("key_file"), None
                    )
                    with open(
                        oci_config.get("security_token_file"), encoding="utf-8"
                    ) as f:
                        st_string = f.read()
                    return oci.auth.signers.SecurityTokenSigner(st_string, pk)

                client_kwargs["config"] = oci.config.from_file(
                    profile_name=values["auth_profile"]
                )
                client_kwargs["signer"] = make_security_token_signer(
                    oci_config=client_kwargs["config"]
                )
            elif values["auth_type"] == OCIAuthType(3).name:
                client_kwargs[
                    "signer"
                ] = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            elif values["auth_type"] == OCIAuthType(4).name:
                client_kwargs[
                    "signer"
                ] = oci.auth.signers.get_resource_principals_signer()
            else:
                raise ValueError("Please provide valid value to auth_type")

            values["client"] = oci.generative_ai_inference.GenerativeAiInferenceClient(
                **client_kwargs
            )

        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex
        except Exception as e:
            raise ValueError(
                "Could not authenticate with OCI client. "
                "Please check if ~/.oci/config exists. "
                "If INSTANCE_PRINCIPLE or RESOURCE_PRINCIPLE is used, "
                "Please check the specified "
                "auth_profile and auth_type are valid."
            ) from e

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    def _get_provider(self) -> str:
        if self.provider is not None:
            provider = self.provider
        else:
            provider = self.model_id.split(".")[0].lower()

        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider derived from model_id: {self.model_id} "
                "Please explicitly pass in the supported provider "
                "when using custom endpoint"
            )
        return provider


class OCIGenAI(LLM, OCIGenAIBase):
    """OCI large language models.

    To authenticate, the OCI client uses the methods described in
    https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm

    The authentifcation method is passed through auth_type and should be one of:
    API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPLE, RESOURCE_PRINCIPLE

    Make sure you have the required policies (profile/roles) to
    access the OCI Generative AI service.
    If a specific config profile is used, you must pass
    the name of the profile (from ~/.oci/config) through auth_profile.

    To use, you must provide the compartment id
    along with the endpoint url, and model id
    as named parameters to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.llms import OCIGenAI

            llm = OCIGenAI(
                    model_id="MY_MODEL_ID",
                    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                    compartment_id="MY_OCID"
                )
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci"

    def _prepare_invocation_object(
        self, prompt: str, stop: Optional[List[str]], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        from oci.generative_ai_inference import models

        oci_llm_request_mapping = {
            "cohere": models.CohereLlmInferenceRequest,
            "meta": models.LlamaLlmInferenceRequest,
        }
        provider = self._get_provider()
        _model_kwargs = self.model_kwargs or {}
        if stop is not None:
            _model_kwargs[self.llm_stop_sequence_mapping[provider]] = stop

        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)

        inference_params = {**_model_kwargs, **kwargs}
        inference_params["prompt"] = prompt
        inference_params["is_stream"] = self.is_stream

        invocation_obj = models.GenerateTextDetails(
            compartment_id=self.compartment_id,
            serving_mode=serving_mode,
            inference_request=oci_llm_request_mapping[provider](**inference_params),
        )

        return invocation_obj

    def _process_response(self, response: Any, stop: Optional[List[str]]) -> str:
        provider = self._get_provider()
        if provider == "cohere":
            text = response.data.inference_response.generated_texts[0].text
        elif provider == "meta":
            text = response.data.inference_response.choices[0].text
        else:
            raise ValueError(f"Invalid provider: {provider}")

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to OCIGenAI generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

               response = llm.invoke("Tell me a joke.")
        """

        invocation_obj = self._prepare_invocation_object(prompt, stop, kwargs)
        response = self.client.generate_text(invocation_obj)
        return self._process_response(response, stop)
