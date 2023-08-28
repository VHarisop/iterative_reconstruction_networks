import hashlib
from typing import Dict


def hash_dict(dictionary: Dict) -> str:
    """Create a hash from a dictionary.

    Args:
        dictionary: The dictionary to hash.

    Returns:
        The hash string.
    """
    dict2hash = ""
    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]
        dict2hash += "%s_%s_" % (str(k), str(v))
    return hashlib.md5(dict2hash.encode()).hexdigest()