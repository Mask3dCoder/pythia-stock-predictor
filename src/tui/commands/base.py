"""
Base Command Class

All TUI commands inherit from this base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any, Callable, Awaitable
from enum import Enum

from src.tui.app import Command, CommandCategory, CommandResult


class CommandHandler(ABC):
    """
    Base class for all command handlers.
    
    Each command should implement this interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Command name."""
        pass
    
    @property
    @abstractmethod
    def aliases(self) -> List[str]:
        """Command aliases (shortcuts)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Command description."""
        pass
    
    @property
    @abstractmethod
    def usage(self) -> str:
        """Command usage example."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> CommandCategory:
        """Command category."""
        pass
    
    @property
    def requires_symbol(self) -> bool:
        """Does this command require a symbol?"""
        return True
    
    @property
    def requires_data(self) -> bool:
        """Does this command require data?"""
        return True
    
    @property
    def min_args(self) -> int:
        """Minimum arguments."""
        return 0
    
    @property
    def max_args(self) -> int:
        """Maximum arguments."""
        return 10
    
    @abstractmethod
    async def execute(self, terminal: 'PythiaTerminal', args: List[str]) -> CommandResult:
        """Execute the command."""
        pass
    
    def get_command(self) -> Command:
        """Get the Command dataclass."""
        async def handler(t: 'PythiaTerminal', a: List[str]) -> CommandResult:
            return await self.execute(t, a)
        
        return Command(
            name=self.name,
            aliases=self.aliases,
            description=self.description,
            usage=self.usage,
            category=self.category,
            handler=handler,
            min_args=self.min_args,
            max_args=self.max_args,
            requires_symbol=self.requires_symbol,
            requires_data=self.requires_data
        )


class DataCommandHandler(CommandHandler):
    """Base class for data-related commands."""
    
    @property
    def category(self) -> CommandCategory:
        return CommandCategory.DATA
    
    @property
    def requires_data(self) -> bool:
        return True


class AnalyticsCommandHandler(CommandHandler):
    """Base class for analytics commands."""
    
    @property
    def category(self) -> CommandCategory:
        return CommandCategory.ANALYTICS


class MLCommandHandler(CommandHandler):
    """Base class for ML commands."""
    
    @property
    def category(self) -> CommandCategory:
        return CommandCategory.ML


class PortfolioCommandHandler(CommandHandler):
    """Base class for portfolio commands."""
    
    @property
    def category(self) -> CommandCategory:
        return CommandCategory.PORTFOLIO
    
    @property
    def requires_symbol(self) -> bool:
        return False


class SystemCommandHandler(CommandHandler):
    """Base class for system commands."""
    
    @property
    def category(self) -> CommandCategory:
        return CommandCategory.SYSTEM
    
    @property
    def requires_symbol(self) -> bool:
        return False
    
    @property
    def requires_data(self) -> bool:
        return False
