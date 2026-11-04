#include "ds4.h"

#define DS4_TOOL_REMINDER_DEPTH_THRESHOLD 30000

bool ds4_should_inject_tool_reminder(size_t conversation_tokens, bool has_tools) {
    if (!has_tools) return false;
    return conversation_tokens >= DS4_TOOL_REMINDER_DEPTH_THRESHOLD;
}

const char *ds4_tool_protocol_reminder_text(void) {
    return "\n[Protocol Reminder: You are in a tools-armed conversation. When an action is required, output a tool call invocation rather than a natural language narrative.]";
}
