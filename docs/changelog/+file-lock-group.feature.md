Added :class:`filelock.FileLockGroup`, an explicit lock group that normalizes a set of lock paths (collapsing
equivalent spellings via the real parent directory and the platform's case rules), deduplicates them, and acquires
them in one canonical order shared by every caller, so two flows submitting the same locks in different orders
serialize instead of deadlocking. One acquisition shares a single timeout budget across all members; if any member
fails, the members that call already took are released in reverse order, and a member held reentrantly by an outer
flow only returns the count the group added. The group is a context manager and never force-releases or deletes a
lock file it does not own.
