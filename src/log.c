/*  gasp -- global address space toolbox

    log -- message logging facility
 */

#include <stdlib.h>
#include <strings.h>
#include "log.h"


/*  log_init() -- set up runtime logging
 */
void log_init(log_t *logger, char *level_env_name)
{
    int level;
    char *cp;

    cp = getenv(level_env_name);
    if (!cp)
        cp = DEFAULT_LOG_LEVEL;
    if (strncasecmp(cp, "debug", 5) == 0)
        level = LOG_LEVEL_DEBUG;
    else if (strncasecmp(cp, "info", 4) == 0)
        level = LOG_LEVEL_INFO;
    else if (strncasecmp(cp, "err", 3) == 0)
        level = LOG_LEVEL_ERR;
    else if (strncasecmp(cp, "critical", 8) == 0)
        level = LOG_LEVEL_CRITICAL;
    else /* if (strncasecmp(cp, "warn", 4) == 0) */
        level = LOG_LEVEL_WARN;

    LOG_SETUP(*logger, level, stdout);
}

