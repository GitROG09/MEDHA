import asyncio
from typing import Dict, List, Optional

from communication.message import Message
from models.schemas import MessageType
from utils.logger import get_logger

logger = get_logger("message_bus")


class MessageBus:
    """Asynchronous message broker for inter-agent communication and challenge exchange."""

    def __init__(self):
        # Per-agent async inboxes
        self._inboxes: Dict[str, asyncio.Queue[Message]] = {}
        # Global audit log of all exchanged messages
        self._history: List[Message] = []
        self._lock = asyncio.Lock()

    def register_agent(self, agent_id: str) -> None:
        """Register an agent and create its private async message inbox."""
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = asyncio.Queue()
            logger.debug(f"Registered agent [{agent_id}] with message bus.")

    def unregister_agent(self, agent_id: str) -> None:
        """Remove agent inbox from message bus."""
        self._inboxes.pop(agent_id, None)

    async def publish(self, message: Message) -> None:
        """Publish a message to a specific agent inbox or broadcast to all registered agents."""
        async with self._lock:
            self._history.append(message)

            if message.recipient_id == "ALL":
                # Broadcast to every agent except the sender
                for agent_id, queue in self._inboxes.items():
                    if agent_id != message.sender_id:
                        await queue.put(message)
            else:
                # Direct message to specific agent inbox
                if message.recipient_id in self._inboxes:
                    await self._inboxes[message.recipient_id].put(message)
                else:
                    logger.warning(
                        f"Target recipient [{message.recipient_id}] not registered on message bus. "
                        f"Message {message.message_id} stored in audit log only."
                    )

        logger.debug(
            f"Message [{message.message_type}] from [{message.sender_id}] "
            f"-> [{message.recipient_id}]: {message.content[:50]}..."
        )

    async def get_message(self, agent_id: str, timeout: Optional[float] = 0.1) -> Optional[Message]:
        """Fetch the next available message for an agent with optional timeout."""
        if agent_id not in self._inboxes:
            return None
        queue = self._inboxes[agent_id]
        try:
            if timeout is not None:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            return await queue.get()
        except (asyncio.TimeoutError, TimeoutError):
            return None

    async def drain_inbox(self, agent_id: str) -> List[Message]:
        """Retrieve and drain all pending messages for an agent."""
        if agent_id not in self._inboxes:
            return []
        queue = self._inboxes[agent_id]
        messages = []
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def get_history(
        self,
        task_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
    ) -> List[Message]:
        """Retrieve filtered message history from the audit log."""
        results = self._history
        if task_id:
            results = [m for m in results if m.task_id == task_id]
        if message_type:
            results = [m for m in results if m.message_type == message_type]
        return list(results)

    async def challenge_finding(
        self,
        sender_id: str,
        target_agent_id: str,
        finding_or_claim: str,
        critique: str,
        task_id: str = "",
    ) -> Message:
        """Convenience method for one agent to formally challenge another agent's assertion."""
        msg = Message(
            sender_id=sender_id,
            recipient_id=target_agent_id,
            task_id=task_id,
            message_type=MessageType.CHALLENGE,
            content=f"CHALLENGE: Assertion '{finding_or_claim}' disputed. Reason: {critique}",
            metadata={"target_assertion": finding_or_claim, "critique": critique},
        )
        await self.publish(msg)
        return msg

    async def share_evidence(
        self,
        sender_id: str,
        content: str,
        evidence_references: List[str],
        recipient_id: str = "ALL",
        task_id: str = "",
    ) -> Message:
        """Convenience method to publish evidence to collaborators."""
        msg = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            task_id=task_id,
            message_type=MessageType.EVIDENCE,
            content=content,
            evidence_references=evidence_references,
        )
        await self.publish(msg)
        return msg

    def clear(self) -> None:
        """Reset the message bus."""
        self._inboxes.clear()
        self._history.clear()

