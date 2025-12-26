#pragma once
// Mock sqlite3.h for IntelliSense
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
#define SQLITE_OK 0
#define SQLITE_DONE 101
#define SQLITE_STATIC ((void(*)(void *))0)
extern const char *sqlite3_errmsg(sqlite3*);
extern int sqlite3_open(const char*, sqlite3**);
extern int sqlite3_close(sqlite3*);
extern int sqlite3_exec(sqlite3*, const char*, int (*)(void*,int,char**,char**), void*, char**);
extern void sqlite3_free(void*);
extern int sqlite3_prepare_v2(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
extern int sqlite3_step(sqlite3_stmt*);
extern int sqlite3_finalize(sqlite3_stmt*);
extern int sqlite3_bind_text(sqlite3_stmt*, int, const char*, int, void(*)(void*));
extern int sqlite3_reset(sqlite3_stmt*);
extern int sqlite3_clear_bindings(sqlite3_stmt*);
