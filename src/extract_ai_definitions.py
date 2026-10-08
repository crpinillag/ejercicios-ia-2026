"""
Extractor multi-proveedor de definiciones formales de Inteligencia Artificial.
Consulta las APIs oficiales de OpenAI, Anthropic y Google Gemini con fallback 
estructurado a respuestas simuladas en caso de no contar con credenciales activas.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any


class LLMProvider(ABC):
    """Interfaz abstracta para los proveedores de modelos de lenguaje."""

    @abstractmethod
    def fetch_definition(self, prompt: str) -> str:
        """Obtiene la definición de IA desde el proveedor correspondiente."""
        pass


class OpenAIProvider(LLMProvider):
    """Cliente para la API de OpenAI."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def fetch_definition(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY no configurada.")
        
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Eres un asistente académico experto en ciencias de la computación."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()


class AnthropicProvider(LLMProvider):
    """Cliente para la API de Anthropic (Claude)."""

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model

    def fetch_definition(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY no configurada.")
        
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=0.2,
            system="Eres un investigador formal en teoría de la computación e inteligencia artificial.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


class GeminiProvider(LLMProvider):
    """Cliente para la API de Google Gemini."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

    def fetch_definition(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no configurada.")
        
        from google import genai
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip()


class SimulatedLLMProvider(LLMProvider):
    """Proveedor simulado estructurado cuando las claves de API no están disponibles."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self._simulated_responses = {
            "OpenAI": (
                "La Inteligencia Artificial es el campo de las ciencias de la computación dedicado al "
                "desarrollo de sistemas orientados a resolver problemas complejos, aprender de grandes "
                "volúmenes de datos e inferir patrones para realizar tareas cognitivas como el razonamiento, "
                "la generación de contenido y la toma autónoma de decisiones."
            ),
            "Anthropic": (
                "La Inteligencia Artificial se define como el estudio y la ingeniería de sistemas "
                "computacionales capaces de percibir su entorno, procesar información de manera probabilística "
                "y ejecutar acciones adaptativas orientadas a metas específicas con un alto grado de autonomía y alineación."
            ),
            "Google Gemini": (
                "La Inteligencia Artificial es la disciplina científica y tecnológica orientada a la creación de "
                "modelos y algoritmos multimodales capaces de procesar datos heterogéneos, comprender el lenguaje natural "
                "y simular capacidades del intelecto humano mediante el aprendizaje continuo."
            )
        }

    def fetch_definition(self, prompt: str) -> str:
        return self._simulated_responses.get(
            self.provider_name, 
            "Definición formal no disponible para el proveedor especificado."
        )


class AIDefinitionExtractorService:
    """Servicio principal encargado de orquestar las consultas y estructurar los resultados."""

    PROMPT_QUERY = (
        "Proporciona tu definición formal, académica y concisa de 'Inteligencia Artificial' "
        "en máximo 3 oraciones, destacando su propósito principal y sus mecanismos fundamentales."
    )

    def __init__(self, use_simulation_fallback: bool = True):
        self.use_simulation_fallback = use_simulation_fallback
        self.providers: Dict[str, LLMProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Inicializa los proveedores reales o sus fallbacks de simulación."""
        provider_classes = [
            ("OpenAI", OpenAIProvider),
            ("Anthropic", AnthropicProvider),
            ("Google Gemini", GeminiProvider)
        ]

        for name, cls in provider_classes:
            try:
                provider_instance = cls()
                # Prueba de presencia de API Key
                if not provider_instance.api_key and self.use_simulation_fallback:
                    print(f"[INFO] API Key para {name} no encontrada. Utilizando respuesta simulada estructurada.")
                    self.providers[name] = SimulatedLLMProvider(name)
                else:
                    self.providers[name] = provider_instance
            except Exception as e:
                if self.use_simulation_fallback:
                    print(f"[WARN] Error inicializando {name} ({e}). Usando modo simulado.")
                    self.providers[name] = SimulatedLLMProvider(name)
                else:
                    raise e

    def execute_extraction(self) -> Dict[str, str]:
        """Ejecuta la extracción de definiciones en todos los proveedores configurados."""
        results = {}
        for provider_name, provider in self.providers.items():
            try:
                print(f"Consultando respuesta formal de: {provider_name}...")
                results[provider_name] = provider.fetch_definition(self.PROMPT_QUERY)
            except Exception as err:
                print(f"[ERROR] Error al consultar {provider_name}: {err}")
                results[provider_name] = f"Error en la extracción: {str(err)}"
        return results


if __name__ == "__main__":
    extractor = AIDefinitionExtractorService(use_simulation_fallback=True)
    definitions = extractor.execute_extraction()

    output_path = "data/ai_definitions_output.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(definitions, f, ensure_ascii=False, indent=2)

    print(f"\nResultados guardados exitosamente en '{output_path}':")
    print(json.dumps(definitions, ensure_ascii=False, indent=2))