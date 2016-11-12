/*  gasp -- global address space toolbox

    log -- message logging
 */

#ifndef LOG_H
#define LOG_H

#include <stdio.h>


enum {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERR,
    LOG_LEVEL_CRITICAL
};

#define DEFAULT_LOG_LEVEL "warn"

typedef struct log_tag {
    int level;
    FILE *f;
} log_t;

#define LOG_SETUP(l, lvl, fp) do {              \
    (l).level = (lvl);                          \
    (l).f = (fp);                               \
} while(0)

#define LOG_DEBUG(l, ...) do {                  \
    if ((l).level <= LOG_LEVEL_DEBUG)           \
        fprintf((l).f, __VA_ARGS__);            \
} while(0)

#define LOG_INFO(l, ...) do {                   \
    if ((l).level <= LOG_LEVEL_INFO)            \
        fprintf((l).f, __VA_ARGS__);            \
} while(0)

#define LOG_WARN(l, ...) do {                   \
    if ((l).level <= LOG_LEVEL_WARN)            \
        fprintf((l).f, __VA_ARGS__);            \
} while(0)

#define LOG_ERR(l, ...) do {                    \
    if ((l).level <= LOG_LEVEL_ERR)             \
        fprintf((l).f, __VA_ARGS__);            \
} while(0)

#define LOG_CRITICAL(l, ...) do {               \
    if ((l).level <= LOG_LEVEL_CRITICAL)        \
        fprintf((l).f, __VA_ARGS__);            \
} while(0)


void log_init(log_t *logger, char *level_env_name);


#endif /* LOG_H */

