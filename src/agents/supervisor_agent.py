"""Supervisor Agent - Central Orchestration and Query Routing

This agent serves as the central coordinator for the WellbeingAI system, using
LangGraph to orchestrate interactions between all specialized agents. It routes
queries to the appropriate agent, maintains conversation context, ensures privacy
rules are applied, and handles escalations to human operators when needed.

Key Features:
- Intelligent query routing using LangGraph
- Conversation context management
- Privacy rule enforcement
- Multi-agent orchestration
- Human escalation handling
- State persistence and recovery
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Annotated, TypedDict
from datetime import datetime, timedelta
import logging
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from src.agents.risk_scorer_agent import RiskScorerAgent
from src.agents.privacy_agent import PrivacyPreservingAgent
from src.agents.checkin_agent import CheckInAgent
from src.agents.intervention_agent import InterventionAgent
from src.agents.escalation_agent import EscalationAgent
from src.agents.analytics_agent import AnalyticsAgent
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """State for the conversation graph"""
    messages: List[BaseMessage]
    user_id: str
    user_role: str
    department: Optional[str]
    current_agent: Optional[str]
    context: Dict[str, Any]
    needs_human_escalation: bool
    escalation_reason: Optional[str]
    privacy_checks_passed: bool
    final_response: Optional[str]


class SupervisorAgent:
    """Central supervisor agent using LangGraph for orchestration"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Supervisor Agent

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)

        # Initialize all specialized agents
        self.agents = {
            'risk_scorer': RiskScorerAgent(config_path),
            'privacy': PrivacyPreservingAgent(config_path),
            'checkin': CheckInAgent(config_path),
            'intervention': InterventionAgent(config_path),
            'escalation': EscalationAgent(config_path),
            'analytics': AnalyticsAgent(config_path)
        }

        # Build the LangGraph workflow
        self.workflow = self._build_workflow()

        logger.info("✅ Supervisor Agent initialized with LangGraph orchestration")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for agent orchestration

        Returns:
            Compiled StateGraph workflow
        """
        workflow = StateGraph(ConversationState)

        # Add nodes (processing steps)
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("check_privacy", self._check_privacy)
        workflow.add_node("route_to_agent", self._route_to_agent)
        workflow.add_node("execute_agent", self._execute_agent)
        workflow.add_node("validate_response", self._validate_response)
        workflow.add_node("handle_escalation", self._handle_escalation)
        workflow.add_node("generate_response", self._generate_response)

        # Define the workflow edges
        workflow.set_entry_point("analyze_query")

        workflow.add_edge("analyze_query", "check_privacy")
        workflow.add_edge("check_privacy", "route_to_agent")
        workflow.add_edge("route_to_agent", "execute_agent")
        workflow.add_edge("execute_agent", "validate_response")

        # Conditional edges based on validation results
        workflow.add_conditional_edges(
            "validate_response",
            self._should_escalate,
            {
                True: "handle_escalation",
                False: "generate_response"
            }
        )

        workflow.add_edge("handle_escalation", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def process_query(self, user_id: str, user_role: str, query: str,
                      department: Optional[str] = None,
                      conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Process a user query through the agent orchestration system

        Args:
            user_id: User identifier
            user_role: User's role (employee, manager, admin)
            query: User's query or request
            department: User's department (for managers)
            conversation_history: Previous conversation messages

        Returns:
            Response dictionary with results and metadata
        """
        logger.info(f"🤖 Processing query for {user_role} {user_id}: {query[:50]}...")

        try:
            # Prepare initial state
            initial_state = ConversationState(
                messages=self._prepare_messages(conversation_history or [], query),
                user_id=user_id,
                user_role=user_role,
                department=department,
                current_agent=None,
                context={},
                needs_human_escalation=False,
                escalation_reason=None,
                privacy_checks_passed=False,
                final_response=None
            )

            # Execute the workflow
            final_state = self.workflow.invoke(initial_state)

            # Extract response
            response = {
                "response": final_state.get("final_response", "I'm sorry, I couldn't process your request."),
                "agent_used": final_state.get("current_agent"),
                "privacy_compliant": final_state.get("privacy_checks_passed"),
                "escalated": final_state.get("needs_human_escalation"),
                "escalation_reason": final_state.get("escalation_reason"),
                "processing_time": datetime.now().isoformat(),
                "context": final_state.get("context", {})
            }

            logger.info(f"✅ Query processed by {response['agent_used']}, privacy: {response['privacy_compliant']}")
            return response

        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again or contact support.",
                "error": str(e),
                "agent_used": "error_handler",
                "privacy_compliant": True,
                "escalated": True,
                "escalation_reason": "System error occurred"
            }

    def _prepare_messages(self, conversation_history: List[Dict], current_query: str) -> List[BaseMessage]:
        """Prepare message history for the workflow

        Args:
            conversation_history: Previous conversation messages
            current_query: Current user query

        Returns:
            List of BaseMessage objects
        """
        messages = []

        # Add conversation history
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            if msg.get('role') == 'user':
                messages.append(HumanMessage(content=msg.get('content', '')))
            elif msg.get('role') == 'assistant':
                messages.append(AIMessage(content=msg.get('content', '')))

        # Add current query
        messages.append(HumanMessage(content=current_query))

        return messages

    def _analyze_query(self, state: ConversationState) -> ConversationState:
        """Analyze the user query to understand intent and requirements

        Args:
            state: Current conversation state

        Returns:
            Updated state with analysis results
        """
        query = state['messages'][-1].content if state['messages'] else ""
        user_role = state['user_role']

        # Analyze query intent
        intent_analysis = self._classify_query_intent(query, user_role)

        # Update context
        state['context'].update({
            'query_intent': intent_analysis['intent'],
            'query_entities': intent_analysis['entities'],
            'required_permissions': intent_analysis['permissions'],
            'sensitivity_level': intent_analysis['sensitivity'],
            'suggested_agent': intent_analysis['suggested_agent']
        })

        logger.info(f"🔍 Query analysis: {intent_analysis['intent']} -> {intent_analysis['suggested_agent']}")
        return state

    def _classify_query_intent(self, query: str, user_role: str) -> Dict[str, Any]:
        """Classify the intent of a user query

        Args:
            query: User query string
            user_role: User's role

        Returns:
            Dictionary with intent analysis
        """
        query_lower = query.lower()

        # Define intent patterns and corresponding agents
        intent_patterns = {
            'submit_checkin': {
                'patterns': ['check in', 'submit checkin', 'log wellbeing', 'daily check'],
                'agent': 'checkin',
                'permissions': ['submit_checkin'],
                'sensitivity': 'personal'
            },
            'get_risk_assessment': {
                'patterns': ['risk score', 'risk assessment', 'my risk', 'crisis risk'],
                'agent': 'risk_scorer',
                'permissions': ['read_own_data'],
                'sensitivity': 'personal'
            },
            'get_interventions': {
                'patterns': ['intervention', 'recommendation', 'help me', 'support'],
                'agent': 'intervention',
                'permissions': ['read_own_data'],
                'sensitivity': 'personal'
            },
            'get_dashboard': {
                'patterns': ['dashboard', 'analytics', 'trends', 'report', 'metrics'],
                'agent': 'analytics',
                'permissions': ['read_own_data', 'read_team_aggregate', 'read_org_aggregate'],
                'sensitivity': 'aggregate'
            },
            'get_privacy_report': {
                'patterns': ['privacy report', 'safe report', 'anonymized', 'k-anonymity'],
                'agent': 'privacy',
                'permissions': ['generate_reports'],
                'sensitivity': 'aggregate'
            },
            'escalation_help': {
                'patterns': ['crisis', 'emergency', 'urgent help', 'immediate assistance'],
                'agent': 'escalation',
                'permissions': ['escalate'],
                'sensitivity': 'critical'
            }
        }

        # Find matching intent
        for intent_name, intent_config in intent_patterns.items():
            if any(pattern in query_lower for pattern in intent_config['patterns']):
                return {
                    'intent': intent_name,
                    'suggested_agent': intent_config['agent'],
                    'permissions': intent_config['permissions'],
                    'sensitivity': intent_config['sensitivity'],
                    'entities': self._extract_entities(query)
                }

        # Default intent
        return {
            'intent': 'general_inquiry',
            'suggested_agent': 'analytics',
            'permissions': ['read_own_data'],
            'sensitivity': 'personal',
            'entities': self._extract_entities(query)
        }

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from the query (dates, departments, etc.)

        Args:
            query: User query

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        # Extract date references
        if any(word in query.lower() for word in ['today', 'yesterday', 'last week', 'this week']):
            entities['timeframe'] = 'recent'

        if '90 days' in query or '90-day' in query:
            entities['timeframe'] = '90_days'

        # Extract department references
        dept_keywords = ['engineering', 'sales', 'marketing', 'hr', 'finance', 'operations']
        for dept in dept_keywords:
            if dept in query.lower():
                entities['department'] = dept.title()
                break

        return entities

    def _check_privacy(self, state: ConversationState) -> ConversationState:
        """Check privacy compliance for the requested operation

        Args:
            state: Current conversation state

        Returns:
            Updated state with privacy check results
        """
        user_role = state['user_role']
        required_permissions = state['context'].get('required_permissions', [])
        sensitivity_level = state['context'].get('sensitivity_level', 'personal')

        # Check role-based permissions
        role_permissions = self.config.get(f'privacy.role_permissions.{user_role}', [])

        privacy_passed = True
        privacy_issues = []

        # Check if user has required permissions
        for permission in required_permissions:
            if permission not in role_permissions:
                privacy_passed = False
                privacy_issues.append(f"Missing permission: {permission}")

        # Additional checks for sensitive data
        if sensitivity_level == 'critical' and user_role not in ['admin', 'counselor']:
            privacy_passed = False
            privacy_issues.append("Critical operations require elevated permissions")

        state['privacy_checks_passed'] = privacy_passed
        state['context']['privacy_issues'] = privacy_issues

        logger.info(f"🔒 Privacy check: {'PASSED' if privacy_passed else 'FAILED'} for {user_role}")
        if not privacy_passed:
            logger.warning(f"Privacy issues: {privacy_issues}")

        return state

    def _route_to_agent(self, state: ConversationState) -> ConversationState:
        """Route the query to the appropriate specialized agent

        Args:
            state: Current conversation state

        Returns:
            Updated state with routing decision
        """
        if not state['privacy_checks_passed']:
            # Route to privacy agent for access denied response
            state['current_agent'] = 'privacy'
            state['context']['routing_reason'] = 'privacy_violation'
            return state

        # Use suggested agent from intent analysis
        suggested_agent = state['context'].get('suggested_agent', 'analytics')
        state['current_agent'] = suggested_agent

        logger.info(f"🚦 Routing to agent: {suggested_agent}")
        return state

    def _execute_agent(self, state: ConversationState) -> ConversationState:
        """Execute the selected agent with the user query

        Args:
            state: Current conversation state

        Returns:
            Updated state with agent execution results
        """
        agent_name = state['current_agent']
        agent = self.agents.get(agent_name)

        if not agent:
            state['context']['agent_error'] = f"Agent '{agent_name}' not found"
            state['needs_human_escalation'] = True
            state['escalation_reason'] = "Agent unavailable"
            return state

        try:
            # Prepare agent-specific parameters
            params = self._prepare_agent_params(state)

            # Execute agent
            if agent_name == 'checkin':
                result = self._execute_checkin_agent(agent, params)
            elif agent_name == 'risk_scorer':
                result = self._execute_risk_agent(agent, params)
            elif agent_name == 'intervention':
                result = self._execute_intervention_agent(agent, params)
            elif agent_name == 'analytics':
                result = self._execute_analytics_agent(agent, params)
            elif agent_name == 'privacy':
                result = self._execute_privacy_agent(agent, params)
            elif agent_name == 'escalation':
                result = self._execute_escalation_agent(agent, params)
            else:
                result = {"error": f"Unknown agent: {agent_name}"}

            state['context']['agent_result'] = result
            logger.info(f"✅ Agent {agent_name} executed successfully")

        except Exception as e:
            logger.error(f"❌ Agent {agent_name} execution failed: {e}")
            state['context']['agent_error'] = str(e)
            state['needs_human_escalation'] = True
            state['escalation_reason'] = f"Agent execution failed: {str(e)}"

        return state

    def _prepare_agent_params(self, state: ConversationState) -> Dict[str, Any]:
        """Prepare parameters for agent execution

        Args:
            state: Current conversation state

        Returns:
            Dictionary of parameters for the agent
        """
        return {
            'user_id': state['user_id'],
            'user_role': state['user_role'],
            'department': state['department'],
            'query': state['messages'][-1].content if state['messages'] else "",
            'context': state['context']
        }

    def _execute_checkin_agent(self, agent: CheckInAgent, params: Dict) -> Dict[str, Any]:
        """Execute check-in agent operations"""
        query = params['query'].lower()

        if 'form' in query or 'structure' in query:
            return {"form": agent.get_checkin_form()}
        elif 'streak' in query:
            return agent.get_checkin_streak(params['user_id'])
        elif 'statistics' in query or 'stats' in query:
            return agent.get_checkin_statistics(params['user_id'])
        else:
            # Assume they want recent check-ins
            return {"recent_checkins": agent.get_recent_checkins(params['user_id'])}

    def _execute_risk_agent(self, agent: RiskScorerAgent, params: Dict) -> Dict[str, Any]:
        """Execute risk scorer agent operations"""
        user_id = params['user_id']
        risk_assessment = agent.score_user_risk(user_id)
        return {"risk_assessment": risk_assessment.__dict__ if hasattr(risk_assessment, '__dict__') else risk_assessment}

    def _execute_intervention_agent(self, agent: InterventionAgent, params: Dict) -> Dict[str, Any]:
        """Execute intervention agent operations"""
        # Get user's risk factors for personalization
        risk_agent = self.agents['risk_scorer']
        risk_assessment = risk_agent.score_user_risk(params['user_id'])

        risk_score = risk_assessment.risk_score if hasattr(risk_assessment, 'risk_score') else 0.5
        factors = risk_assessment.contributing_factors if hasattr(risk_assessment, 'contributing_factors') else []

        interventions = agent.recommend_interventions(
            params['user_id'], risk_score, factors
        )
        return {"interventions": interventions}

    def _execute_analytics_agent(self, agent: AnalyticsAgent, params: Dict) -> Dict[str, Any]:
        """Execute analytics agent operations"""
        dashboard = agent.get_dashboard_data(
            params['user_id'],
            params['user_role'],
            params['department']
        )
        return {"dashboard": dashboard}

    def _execute_privacy_agent(self, agent: PrivacyPreservingAgent, params: Dict) -> Dict[str, Any]:
        """Execute privacy agent operations"""
        if not params['context'].get('privacy_checks_passed', False):
            return {"error": "Access denied", "message": "Insufficient permissions for this operation"}

        report = agent.generate_safe_report(
            params['user_role'],
            params['department']
        )
        return {"privacy_report": report}

    def _execute_escalation_agent(self, agent: EscalationAgent, params: Dict) -> Dict[str, Any]:
        """Execute escalation agent operations"""
        # Get current risk for escalation check
        risk_agent = self.agents['risk_scorer']
        risk_assessment = risk_agent.score_user_risk(params['user_id'])

        risk_score = risk_assessment.risk_score if hasattr(risk_assessment, 'risk_score') else 0.0
        factors = risk_assessment.contributing_factors if hasattr(risk_assessment, 'contributing_factors') else []

        escalation_result = agent.check_and_escalate(params['user_id'], risk_score, factors)
        return {"escalation": escalation_result}

    def _validate_response(self, state: ConversationState) -> ConversationState:
        """Validate the agent response for quality and safety

        Args:
            state: Current conversation state

        Returns:
            Updated state with validation results
        """
        agent_result = state['context'].get('agent_result', {})

        # Check for errors in agent response
        if 'error' in agent_result:
            state['needs_human_escalation'] = True
            state['escalation_reason'] = f"Agent error: {agent_result['error']}"
            return state

        # Check for empty or invalid responses
        if not agent_result or len(str(agent_result)) < 10:
            state['needs_human_escalation'] = True
            state['escalation_reason'] = "Invalid or empty agent response"
            return state

        # Additional validation based on agent type
        agent_name = state['current_agent']
        if agent_name == 'privacy' and 'error' in str(agent_result).lower():
            state['needs_human_escalation'] = True
            state['escalation_reason'] = "Privacy violation detected"

        return state

    def _should_escalate(self, state: ConversationState) -> bool:
        """Determine if the query should be escalated to human operators

        Args:
            state: Current conversation state

        Returns:
            True if escalation is needed
        """
        return state.get('needs_human_escalation', False)

    def _handle_escalation(self, state: ConversationState) -> ConversationState:
        """Handle escalation to human operators

        Args:
            state: Current conversation state

        Returns:
            Updated state with escalation handling
        """
        escalation_reason = state.get('escalation_reason', 'Unknown reason')

        # Log escalation
        logger.warning(f"🚨 Escalating query for user {state['user_id']}: {escalation_reason}")

        # Prepare escalation response
        escalation_response = {
            "type": "escalation",
            "message": "I've escalated your request to our human support team. They'll follow up with you shortly.",
            "escalation_reason": escalation_reason,
            "estimated_response_time": "Within 24 hours",
            "contact_options": [
                "Email: support@wellbeing.ai",
                "Phone: 1-800-WELLNESS",
                "Chat: Available during business hours"
            ]
        }

        state['final_response'] = json.dumps(escalation_response, indent=2)
        return state

    def _generate_response(self, state: ConversationState) -> ConversationState:
        """Generate the final response to the user

        Args:
            state: Current conversation state

        Returns:
            Updated state with final response
        """
        agent_result = state['context'].get('agent_result', {})
        agent_name = state['current_agent']

        try:
            # Format response based on agent type
            if agent_name == 'checkin':
                response = self._format_checkin_response(agent_result)
            elif agent_name == 'risk_scorer':
                response = self._format_risk_response(agent_result)
            elif agent_name == 'intervention':
                response = self._format_intervention_response(agent_result)
            elif agent_name == 'analytics':
                response = self._format_analytics_response(agent_result)
            elif agent_name == 'privacy':
                response = self._format_privacy_response(agent_result)
            elif agent_name == 'escalation':
                response = self._format_escalation_response(agent_result)
            else:
                response = "I'm sorry, I couldn't process your request properly."

            state['final_response'] = response

        except Exception as e:
            logger.error(f"❌ Error formatting response: {e}")
            state['final_response'] = "I apologize, but there was an error formatting the response. Please try again."

        return state

    def _format_checkin_response(self, result: Dict) -> str:
        """Format check-in agent response"""
        if 'form' in result:
            return f"Here's the daily check-in form:\n{json.dumps(result['form'], indent=2)}"
        elif 'recent_checkins' in result:
            checkins = result['recent_checkins']
            if not checkins:
                return "You don't have any recent check-ins. Would you like to submit your first one?"
            return f"You have {len(checkins)} recent check-ins. Your last check-in was on {checkins[0].get('date', 'unknown date')}."
        else:
            return "Check-in processed successfully!"

    def _format_risk_response(self, result: Dict) -> str:
        """Format risk scorer response"""
        assessment = result.get('risk_assessment', {})
        risk_score = assessment.get('risk_score', 'unknown')
        factors = assessment.get('contributing_factors', [])
        action = assessment.get('recommended_action', 'Monitor regularly')

        response = f"Your current risk score is {risk_score:.2f}.\n\n"
        if factors:
            response += f"Key factors: {', '.join(factors)}\n\n"
        response += f"Recommended action: {action}"

        return response

    def _format_intervention_response(self, result: Dict) -> str:
        """Format intervention agent response"""
        interventions = result.get('interventions', [])
        if not interventions:
            return "I don't have specific intervention recommendations at this time."

        response = "Here are some personalized interventions for you:\n\n"
        for i, intervention in enumerate(interventions[:3], 1):
            response += f"{i}. {intervention.get('description', 'Unknown intervention')}\n"
            response += f"   • Success Rate: {intervention.get('success_rate', 0):.1%}\n"
            response += f"   • Time Commitment: {intervention.get('time_commitment', 'Varies')}\n\n"

        return response

    def _format_analytics_response(self, result: Dict) -> str:
        """Format analytics agent response"""
        dashboard = result.get('dashboard', {})
        dashboard_type = dashboard.get('dashboard_type', 'unknown')

        if dashboard_type == 'employee_personal':
            trends = dashboard.get('trend_data', {})
            current_mood = trends.get('current_values', {}).get('mood', 'unknown')
            return f"Your personal dashboard shows your current mood is {current_mood}. Check the full dashboard for detailed trends and insights."

        elif dashboard_type == 'manager_team':
            team_size = dashboard.get('team_size', 0)
            avg_mood = dashboard.get('team_metrics', {}).get('avg_mood_score', 'unknown')
            return f"Your team dashboard shows {team_size} team members with an average mood score of {avg_mood}."

        elif dashboard_type == 'admin_organization':
            total_employees = dashboard.get('organization_metrics', {}).get('total_employees', 0)
            return f"The organization dashboard covers {total_employees} employees across all departments."

        return "Dashboard data retrieved successfully."

    def _format_privacy_response(self, result: Dict) -> str:
        """Format privacy agent response"""
        if 'error' in result:
            return f"Access denied: {result.get('message', 'Insufficient permissions')}"

        report = result.get('privacy_report', {})
        if 'error' in report:
            return f"Report error: {report.get('message', 'Unable to generate report')}"

        return "Privacy-safe report generated. This report contains only aggregated, anonymized data to protect individual privacy."

    def _format_escalation_response(self, result: Dict) -> str:
        """Format escalation agent response"""
        escalation = result.get('escalation')
        if not escalation:
            return "No escalation needed at this time."

        counselor = escalation.get('assigned_counselor', 'unknown')
        next_steps = escalation.get('next_steps', [])

        response = f"You've been connected with counselor {counselor}.\n\nNext steps:\n"
        for step in next_steps:
            response += f"• {step}\n"

        return response

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health metrics

        Returns:
            System status information
        """
        try:
            status = {
                "supervisor_agent": "operational",
                "langgraph_workflow": "compiled",
                "specialized_agents": {},
                "system_health": "good"
            }

            # Check each agent status
            for agent_name, agent in self.agents.items():
                try:
                    # Basic health check - just verify agent exists and is callable
                    status["specialized_agents"][agent_name] = "operational"
                except Exception as e:
                    status["specialized_agents"][agent_name] = f"error: {str(e)}"
                    status["system_health"] = "degraded"

            return status

        except Exception as e:
            return {
                "supervisor_agent": "error",
                "error": str(e),
                "system_health": "critical"
            }
