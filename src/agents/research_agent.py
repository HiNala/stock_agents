import logging
import os
from typing import Dict, Any
import pandas as pd
from src.llm.base_llm_client import BaseLLMClient
from src.config.settings import LOG_DIR

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'research_agent.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with LLM client."""
        self.llm_client = BaseLLMClient("research_agent")
        logger.info("Research agent initialized with LLM client")

    def update_llm_config(self, new_config: Dict[str, Any]) -> None:
        """Update the LLM configuration for the research agent."""
        self.llm_client.update_config(new_config)
        logger.info(f"Updated LLM configuration: {new_config}")

    async def generate_research_report(self, universe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate a research report for the given universe."""
        try:
            # Prepare the prompt with universe data
            prompt = self._prepare_research_prompt(universe_data)
            
            # Generate the research report using LLM
            report = await self.llm_client.generate(prompt)
            
            # Parse and structure the report
            structured_report = self._parse_research_report(report)
            
            logger.info("Successfully generated research report")
            return structured_report
            
        except Exception as e:
            logger.error(f"Error generating research report: {str(e)}")
            raise

    def _prepare_research_prompt(self, universe_data: Dict[str, pd.DataFrame]) -> str:
        """Prepare the research prompt with universe data."""
        # Extract key metrics from the universe data
        metrics = self._extract_universe_metrics(universe_data)
        
        prompt = f"""
        Analyze the following stock universe and generate a comprehensive research report:
        
        Universe Metrics:
        {metrics}
        
        Please provide:
        1. Overall market analysis
        2. Key trends and patterns
        3. Risk factors
        4. Investment opportunities
        5. Recommendations
        """
        
        return prompt

    def _extract_universe_metrics(self, universe_data: Dict[str, pd.DataFrame]) -> str:
        """Extract key metrics from the universe data."""
        metrics = []
        for ticker, data in universe_data.items():
            metrics.append(f"\n{ticker}:")
            metrics.append(f"  - Current Price: {data['Close'].iloc[-1]:.2f}")
            metrics.append(f"  - 30-day Volatility: {data['Close'].pct_change().std() * 100:.2f}%")
            metrics.append(f"  - 30-day Return: {((data['Close'].iloc[-1] / data['Close'].iloc[-30]) - 1) * 100:.2f}%")
        
        return "\n".join(metrics)

    def _parse_research_report(self, report: str) -> Dict[str, Any]:
        """Parse the LLM-generated report into a structured format."""
        # This is a simplified parser - you might want to implement a more robust one
        sections = {
            "market_analysis": "",
            "trends": "",
            "risk_factors": "",
            "opportunities": "",
            "recommendations": ""
        }
        
        current_section = None
        for line in report.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if "1." in line:
                current_section = "market_analysis"
            elif "2." in line:
                current_section = "trends"
            elif "3." in line:
                current_section = "risk_factors"
            elif "4." in line:
                current_section = "opportunities"
            elif "5." in line:
                current_section = "recommendations"
            elif current_section:
                sections[current_section] += line + "\n"
        
        return sections 