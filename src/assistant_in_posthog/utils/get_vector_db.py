import threading

from langchain_chroma import Chroma


class VectorDB:
    """Holds one shared Chroma instance for the whole process."""

    _instance = None
    _dbDirectory = None
    _lock = threading.Lock()

    @classmethod
    def getInstance(cls, dbDirectory, embedding):
        """
        Return the shared Chroma instance, creating it on the first call.
        Later calls return the same instance; asking for a different
        dbDirectory raises instead of silently returning the wrong db.
        """
        if cls._instance is None:
            with cls._lock:
                # Check again: another thread may have created it while we waited
                if cls._instance is None:
                    cls._instance = Chroma(persist_directory=dbDirectory, embedding_function=embedding)
                    cls._dbDirectory = dbDirectory

        if dbDirectory != cls._dbDirectory:
            raise ValueError(
                f"VectorDB already uses {cls._dbDirectory!r}, cannot switch to {dbDirectory!r}"
            )
        return cls._instance


def getDb(dbDirectory, embedding):
    """Shortcut for VectorDB.getInstance()."""
    return VectorDB.getInstance(dbDirectory, embedding)
