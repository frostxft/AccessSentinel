from __future__ import annotations


from pydantic import BaseModel, ConfigDict, Field, field_validator


class IdentitySummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    user_id: str
    username: str
    email: str
    department: str
    source_system: str
    score: int
    tier: str
    anomaly_types: list[str] = Field(default_factory=list)
    mitre_technique: str = ""
    primary_rule: str = ""
    suppressed_rule_count: int = 0
    blast_radius_applied: bool = False
    cluster_assignment: str = ""


class IdentityDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    user_id: str
    username: str
    email: str
    department: str
    source_system: str
    score: int
    tier: str
    anomaly_types: list[str] = Field(default_factory=list)
    mitre_technique: str = ""
    primary_rule: str = ""
    suppressed_rule_count: int = 0
    blast_radius_applied: bool = False
    account_type: str
    employment_status: str
    mfa_enabled: bool
    sso_linked: bool
    last_login: str | None = None
    created_at: str | None = None
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    systems_count: int = 0
    off_hours_access_pct: float = 0.0
    is_privileged: bool = False
    risk_narrative: str = ""
    contributing_factors: list[str] = Field(default_factory=list)
    suppressed_factors: list[str] = Field(default_factory=list)
    context_signals: list[dict] = Field(default_factory=list)
    mitre_techniques: list[dict] = Field(default_factory=list)
    remediation_actions: list[dict] = Field(default_factory=list)
    behavior_zscore: float = 0.0
    confidence: float = 0.0
    cluster_assignment: str = ""
    peer_deviation_score: float = 0.0
    sequence_risk: dict | None = None


class RuleResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    rule_id: str
    severity: str
    triggered: bool
    evidence_text: str
    suppressed_by: str | None = None


class ContextSignalSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    signal_type: str
    explanation: str
    confidence: float
    score_adjustment: int
    rules_suppressed: list[str] = Field(default_factory=list)
    requires_followup: bool = False


class MitreTechniqueSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    technique_id: str
    name: str
    tactic: str
    url: str
    triggered_by_rule: str


class RemediationActionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    priority: int
    action_type: str
    target: str
    human_readable_description: str
    machine_actionable_command: str
    estimated_risk_reduction: int
    expected_resolution_hours: int
    requires_approval: bool


class AccessRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    department: str
    job_title: str
    resource: str
    action: str


class AccessDecisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    decision: str
    confidence: float
    peer_comparison_note: str


class PaginatedResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    items: list
    total: int
    page: int
    page_size: int
    pages: int


class EvaluationReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    overall_f1: float
    macro_f1: float
    weighted_f1: float
    precision_by_class: dict
    recall_by_class: dict
    f1_by_class: dict
    false_positive_rate: float
    false_positive_rate_by_dept: dict
    confusion_matrix_data: list[list[int]]
    source: str = "sample_data"
    labels_source: str = "sample_data"
    evaluation_status: str = "ok"
    label_match_count: int = 0


class ClusterSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    cluster_id: str
    label: str
    user_count: int
    avg_risk_score: float
    dominant_resources: list[str] = Field(default_factory=list)
    dominant_actions: list[str] = Field(default_factory=list)
    outlier_count: int = 0


class HealthResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    status: str
    models_loaded: bool
    baseline_loaded: bool
    version: str


class ScanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    message: str
    leaderboard: list[IdentitySummary] = Field(default_factory=list)
    total: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    f1_score: float | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    detail: str
    request_id: str


class BlastRadiusReport(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    identity_id: str
    systems_at_risk: list[str] = Field(default_factory=list)
    downstream_users: list[str] = Field(default_factory=list)
    sensitive_resources: list[str] = Field(default_factory=list)
    estimated_impact_score: int = 0
    narrative: str = ""


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    identity_id: str
    original_decision: str
    corrected_decision: str
    correction_reason: str

    @field_validator("corrected_decision")
    @classmethod
    def validate_decision(cls, value: str) -> str:
        if value not in {"APPROVE", "DENY", "REVIEW"}:
            raise ValueError("corrected_decision must be one of APPROVE, DENY, REVIEW")
        return value


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    feedback_id: str
    message: str
