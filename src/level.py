#
# Copyright (C) Lextend - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Cam Tu Nguyen, July 2016
#


class LevelDetect():
    def __init__(self, langid):
        self.langid = langid

    def detect_level(self, wordlist, extras):
        """
        Detect readability level for a wordlist.

        Parameters:
        ---
        wordlist: list of words
        extras: dictionary, mapping from auxiliary data to its values
        """
        return 0
