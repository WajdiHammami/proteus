
from typing import Type
from pydantic import BaseModel
from src.ports.extraction_port import ExtractionPort
from enum import Enum

class ExtractionTier(Enum):
    HEURISTIC = "heuristic"       # DOM / Regex ($0, instant)
    SLM = "slm"                   # Small Language Model (Fast, low cost)
    CONSTRAINED_LLM = "llm"       # Heavy LLM with strict grammar (High accuracy, costly)




class ExtractionRouter:
    """
    Decides which extraction technique/adapter to invoke based on 
    source characteristics, past success rates, or fallback triggers.
    """
    def __init__(
        self, 
        heuristic_adapter: ExtractionPort,
        slm_adapter: ExtractionPort,
        llm_adapter: ExtractionPort
    ):
        self.extractors = {
            ExtractionTier.HEURISTIC: heuristic_adapter,
            ExtractionTier.SLM: slm_adapter,
            ExtractionTier.CONSTRAINED_LLM: llm_adapter,
        }

        self.tier_sequence = [
            ExtractionTier.HEURISTIC,
            ExtractionTier.SLM,
            ExtractionTier.CONSTRAINED_LLM
        ]
        

    def route_and_extract(
        self, 
        cleaned_text: str, 
        dynamic_model: Type[BaseModel],
        forced_tier: ExtractionTier = None
    ) -> tuple[BaseModel, ExtractionTier]:
        """
        Executes extraction with a cascading fallback strategy:
        Tries cheaper tiers first, escalates if validation or confidence drops.
        """

        tier_sequence = self.tier_sequence

        if forced_tier:
            tier_sequence = [forced_tier]

        last_error = None
        for tier in tier_sequence:
            extractor = self.extractors[tier]
            try:
                print(f"[{tier.value.upper}] Attempting extraction...")
                result = extractor.extract(cleaned_text, dynamic_model)
                
                # Optional: Add business logic check (e.g., did it miss required fields?)
                if self._is_extraction_complete(result):
                    return result, tier
                else:
                    print(f"[{tier.value.upper}] Incomplete data extracted, escalating...")
                    
            except Exception as e:
                print(f"[{tier.value.upper}] Failed with error: {e}")
                last_error = e
                continue

        raise RuntimeError(f"All extraction tiers failed. Last error: {last_error}")

    def _is_extraction_complete(self, result: BaseModel) -> bool:
        """Checks if critical fields are populated or if fallback is needed."""
        # Example check: ensure model fields aren't all default/None values
        data_dict = result.model_dump()
        filled_fields = sum(1 for v in data_dict.values() if v is not None)
        total_fields = len(data_dict)
        
        # If less than half the fields were found, consider it incomplete
        return (filled_fields / total_fields) >= 0.5