"""Multi-agent heart disease diagnosis system (UCI Heart Disease dataset).

This package adapts the CrewAI `medical_diagnosis` template
(https://github.com/RoxanaSz/medical_diagnosis) to the cardiovascular domain.
Each specialist agent reasons over a clinically meaningful subset of the
dataset features and emits a structured JSON diagnosis. A coordinator agent
then aggregates the agent outputs into a final decision.
"""

__version__ = "0.2.0"
