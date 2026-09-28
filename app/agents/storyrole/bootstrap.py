"""StoryRole Agent bootstrap and A2A cards."""

from __future__ import annotations

from app.agents.a2a.protocol import A2AAgentCard
from app.agents.a2a.registry import registry
from app.agents.storyrole.character_agent import CharacterConversationAgent
from app.agents.storyrole.character_resolver import CharacterResolverAgent
from app.agents.storyrole.consistency_guard import ConsistencyGuardAgent
from app.agents.storyrole.context_curator import ContextCuratorAgent
from app.agents.storyrole.memory_agent import MemoryDecisionAgent
from app.agents.storyrole.nuwa_profiler import NuwaProfilerAgent
from app.agents.storyrole.period_agent import CharacterPeriodAnalysisAgent
from app.agents.storyrole.deep_planner import DeepQuestionPlannerAgent
from app.agents.storyrole.web_search_agent import StoryRoleWebSearchAgent
from app.agents.storyrole.evidence_analyst import EvidenceAnalysisAgent
from app.agents.storyrole.role_cognition_agent import RoleCognitionAgent
from app.agents.storyrole.query_planner import QueryPlannerAgent
from app.agents.storyrole.reasoning_agent import CharacterReasoningAgent
from app.agents.storyrole.relationship_agent import RelationshipStateAgent
from app.agents.storyrole.timeline_guard import TimelineGuardAgent

_resolver = CharacterResolverAgent()
_profiler = NuwaProfilerAgent()
_periods = CharacterPeriodAnalysisAgent()
_deep_planner = DeepQuestionPlannerAgent()
_web_search = StoryRoleWebSearchAgent()
_evidence_analyst = EvidenceAnalysisAgent()
_role_cognition = RoleCognitionAgent()
_character = CharacterConversationAgent()
_planner = QueryPlannerAgent()
_curator = ContextCuratorAgent()
_reasoner = CharacterReasoningAgent()
_guard = ConsistencyGuardAgent()
_relationship = RelationshipStateAgent()
_timeline = TimelineGuardAgent()
_memory = MemoryDecisionAgent()


_registered = False

def register_storyrole_agents() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    registry.register(
        "character-resolver",
        A2AAgentCard(
            name="StoryRole Character Resolver",
            description="Checks whether a user-specified character exists in an imported novel.",
            skill_id="resolve-character",
            skill_name="Resolve Character",
            skill_description="Find a character by exact name and return evidence and confidence.",
            url="/api/v1/a2a/character-resolver",
        ),
        _resolver.resolve,
    )
    registry.register(
        "character-period-analysis",
        A2AAgentCard(
            name="StoryRole Character Period Analysis",
            description="Investigates indexed character evidence and recommends temporal persona periods.",
            skill_id="analyze-character-periods",
            skill_name="Analyze Character Periods",
            skill_description="Decide whether a character is stable or needs evidence-backed periods.",
            url="/api/v1/a2a/character-period-analysis",
        ),
        _periods.analyze,
    )
    registry.register(
        "nuwa-profiler",
        A2AAgentCard(
            name="StoryRole Nuwa Profiler",
            description="Builds and persists a structured character profile from novel evidence.",
            skill_id="profile-character",
            skill_name="Profile Character",
            skill_description="Distill personality, beliefs, speech, relationships and boundaries.",
            url="/api/v1/a2a/nuwa-profiler",
        ),
        _profiler.profile,
    )
    registry.register(
        "deep-question-planner",
        A2AAgentCard(
            name="StoryRole Deep Question Planner",
            description="Decomposes a difficult character question into evidence-oriented subqueries.",
            skill_id="plan-deep-question",
            skill_name="Plan Deep Question",
            skill_description="Build bounded retrieval queries for deep character answers.",
            url="/api/v1/a2a/deep-question-planner",
        ),
        _deep_planner.plan,
    )
    registry.register(
        "evidence-analysis",
        A2AAgentCard(
            name="StoryRole Evidence Analysis",
            description="Separates novel-supported facts from character inference.",
            skill_id="analyze-evidence",
            skill_name="Analyze Evidence",
            skill_description="Synthesize bounded evidence without exposing chain of thought.",
            url="/api/v1/a2a/evidence-analysis",
        ),
        _evidence_analyst.analyze,
    )
    registry.register(
        "role-cognition",
        A2AAgentCard(
            name="StoryRole Role Cognition",
            description="Chooses the character's stance and response action for a deep turn.",
            skill_id="decide-role-cognition",
            skill_name="Decide Role Cognition",
            skill_description="Preserve values, emotion and relationship posture before expression.",
            url="/api/v1/a2a/role-cognition",
        ),
        _role_cognition.decide,
    )
    registry.register(
        "web-search",
        A2AAgentCard("StoryRole Web Search", "Retrieves bounded external evidence through Tavily.", "web-search", "Web Search", "Search external sources without changing the character timeline.", "/api/v1/a2a/web-search"),
        _web_search.search,
    )
    registry.register(
        "query-planner",
        A2AAgentCard("StoryRole Query Planner", "Plans retrieval and response needs for one turn.", "plan-query", "Plan Query", "Build a structured conversation execution plan.", "/api/v1/a2a/query-planner"),
        _planner.plan,
    )
    registry.register(
        "context-curator",
        A2AAgentCard("StoryRole Context Curator", "Selects only relevant profile, memory and novel evidence.", "curate-context", "Curate Context", "Assemble bounded context for a response.", "/api/v1/a2a/context-curator"),
        _curator.curate,
    )
    registry.register(
        "character-reasoning",
        A2AAgentCard("StoryRole Character Reasoning", "Produces a bounded internal response decision.", "reason-character", "Reason As Character", "Choose emotion, action and response constraints.", "/api/v1/a2a/character-reasoning"),
        _reasoner.reason,
    )
    registry.register(
        "consistency-guard",
        A2AAgentCard("StoryRole Consistency Guard", "Reviews a draft for persona and evidence drift.", "check-consistency", "Check Consistency", "Review and minimally revise a character answer.", "/api/v1/a2a/consistency-guard"),
        _guard.review,
    )
    registry.register(
        "relationship-state",
        A2AAgentCard("StoryRole Relationship State", "Resolves character relationship context.", "get-relationship-state", "Get Relationship", "Return relationship context for the turn.", "/api/v1/a2a/relationship-state"),
        _relationship.resolve,
    )
    registry.register(
        "relationship-update",
        A2AAgentCard("StoryRole Relationship Update", "Persists a reviewed relationship state change.", "update-relationship", "Update Relationship", "Persist a bounded relationship state.", "/api/v1/a2a/relationship-update"),
        _relationship.update_user_state,
    )
    registry.register(
        "timeline-guard",
        A2AAgentCard("StoryRole Timeline Guard", "Checks knowledge and spoiler boundaries.", "check-timeline", "Check Timeline", "Prevent unsupported future knowledge.", "/api/v1/a2a/timeline-guard"),
        _timeline.check,
    )
    registry.register(
        "memory-decision",
        A2AAgentCard("StoryRole Memory Decision", "Decides whether a turn contains durable user memory.", "decide-memory", "Decide Memory", "Classify durable memory candidates.", "/api/v1/a2a/memory-decision"),
        _memory.decide,
    )
    registry.register(
        "character-conversation",
        A2AAgentCard(
            name="StoryRole Character Conversation",
            description="Runs an immersive conversation with a persisted user-created character.",
            skill_id="chat-with-character",
            skill_name="Chat With Character",
            skill_description="Use profile, novel evidence and conversation history to answer in character.",
            url="/api/v1/a2a/character-conversation",
        ),
        _character.reply,
    )
