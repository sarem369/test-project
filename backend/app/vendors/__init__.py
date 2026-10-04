"""Multi-vendor defensive security update connectors."""

from .registry import list_vendors, sync_all, sync_vendor

__all__ = ["list_vendors", "sync_all", "sync_vendor"]
