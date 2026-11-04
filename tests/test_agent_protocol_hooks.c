#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include "ds4.h"

int main(void) {
    // Under 30k tokens: never inject reminder
    assert(ds4_should_inject_tool_reminder(10000, true) == false);
    assert(ds4_should_inject_tool_reminder(29999, true) == false);

    // Over 30k tokens without tools: do not inject reminder
    assert(ds4_should_inject_tool_reminder(30000, false) == false);
    assert(ds4_should_inject_tool_reminder(35000, false) == false);

    // Over 30k tokens WITH tools armed: MUST inject reminder
    assert(ds4_should_inject_tool_reminder(30000, true) == true);
    assert(ds4_should_inject_tool_reminder(35000, true) == true);

    const char *reminder = ds4_tool_protocol_reminder_text();
    assert(reminder != NULL);
    assert(strlen(reminder) > 10);

    printf("test_agent_protocol_hooks: PASS\n");
    return 0;
}
