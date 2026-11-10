from src.core.router import ExtractionRouter
from src.core.schema_loader import load_schema_as_pydantic




#TODO: fetcher, normalizer are not yet defined
class ExtractionPipeline:
    def __init__(self, fetcher, normalizer, router: ExtractionRouter, schema_path: str):
        self.fetcher = fetcher
        self.normalizer = normalizer
        self.router = router
        self.dynamic_model = load_schema_as_pydantic(schema_path)

    def run(self, source_uri: str):
        raw = self.fetcher.fetch(source_uri)
        normalized = self.normalizer.normalize(raw)
        
        # Router handles strategy selection and cascading fallbacks automatically
        extracted_data, used_tier = self.router.route_and_extract(normalized.content, self.dynamic_model)
        
        print(f"Successfully extracted using tier: {used_tier.value}")
        return extracted_data