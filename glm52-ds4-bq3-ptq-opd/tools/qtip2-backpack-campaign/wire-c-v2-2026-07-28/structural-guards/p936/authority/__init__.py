"""Structural content-addressed authority and lifecycle guards."""

from .authority_guard import AuthorityStore, GuardViolation

__all__ = ["AuthorityStore", "GuardViolation"]
