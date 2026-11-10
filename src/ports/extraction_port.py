from abc import ABC, abstractmethod
from typing import Type
from pydantic import BaseModel




class ExtractionPort(ABC):
    """
    Abstract Port defining the contract for parsing cleaned text into 
    strongly-typed, schema-compliant data structures.
    """

    @abstractmethod
    def extract(self, cleaned_text: str, dynamic_model: Type[BaseModel]) -> BaseModel:
        """
        Extracts structured data from unstructured text based on a provided Pydantic model.

        Args:
            cleaned_text (str): The normalized text/markdown content of the document.
            dynamic_model (Type[BaseModel]): The runtime Pydantic model compiled from the user's YAML schema.

        Returns:
            BaseModel: An instantiated, validated Pydantic object conforming exactly to the dynamic model.
        """
        pass