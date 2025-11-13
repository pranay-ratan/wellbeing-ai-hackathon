"""Data schemas for Delta Lake tables"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class DailyCheckIn:
    """Daily mental health check-in record"""
    user_id: str
    check_in_date: datetime
    mood_score: int  # 1-10
    stress_level: int  # 1-10
    energy_level: int  # 1-10
    social_contact_rating: int  # 1-5
    sleep_hours: float  # 4-9
    isolation_flag: bool
    concerns: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'user_id': self.user_id,
            'check_in_date': self.check_in_date.isoformat(),
            'mood_score': self.mood_score,
            'stress_level': self.stress_level,
            'energy_level': self.energy_level,
            'social_contact_rating': self.social_contact_rating,
            'sleep_hours': self.sleep_hours,
            'isolation_flag': self.isolation_flag,
            'concerns': self.concerns
        }


@dataclass
class RiskScore:
    """Mental health risk assessment result"""
    user_id: str
    assessment_date: datetime
    risk_score: float  # 0-1
    contributing_factors: List[str]
    recommended_action: str
    model_version: str
    confidence_score: float
    reasoning: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'user_id': self.user_id,
            'assessment_date': self.assessment_date.isoformat(),
            'risk_score': self.risk_score,
            'contributing_factors': self.contributing_factors,
            'recommended_action': self.recommended_action,
            'model_version': self.model_version,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning
        }


@dataclass
class Intervention:
    """Recommended intervention for user"""
    intervention_id: str
    user_id: str
    recommended_date: datetime
    intervention_type: str
    description: str
    success_rate: float
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        return {
            'intervention_id': self.intervention_id,
            'user_id': self.user_id,
            'recommended_date': self.recommended_date.isoformat(),
            'intervention_type': self.intervention_type,
            'description': self.description,
            'success_rate': self.success_rate
        }


@dataclass
class Escalation:
    """Crisis escalation record"""
    escalation_id: str
    user_id: str
    created_date: datetime
    severity_level: str  # high, critical
    assigned_counselor: Optional[str]
    status: str  # pending, assigned, in_progress, resolved
    resolution_date: Optional[datetime] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'escalation_id': self.escalation_id,
            'user_id': self.user_id,
            'created_date': self.created_date.isoformat(),
            'severity_level': self.severity_level,
            'assigned_counselor': self.assigned_counselor,
            'status': self.status,
            'resolution_date': self.resolution_date.isoformat() if self.resolution_date else None,
            'notes': self.notes
        }


@dataclass
class AuditLog:
    """Audit trail for data access"""
    log_id: str
    timestamp: datetime
    user_id: str
    role: str
    action: str
    resource: str
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'role': self.role,
            'action': self.action,
            'resource': self.resource,
            'details': self.details
        }
