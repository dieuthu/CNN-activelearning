#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
The following functions implement the New Dale-Chall Readability level:
4.9 and Below   Grade 4 and Below: level 0-1
5.0 to 5.9  Grades 5 - 6: level 2
6.0 to 6.9  Grades 7 - 8: level 2
7.0 to 7.9  Grades 9 - 10: level 3
8.0 to 8.9  Grades 11 - 12: level 4
9.0 to 9.9  Grades 13 - 15 (College): level 5
10 and Above    Grades 16 and Above (College Graduate)
We map these levels to our learning levels: 0 - 5


For Chinese:
4.9 and Below   Grade 4 and Below: level 0-1
5.0 to 5.9  Grades 5 - 6: level 2
6.0 to 6.9  Grades 7 - 8: level 3
7.0 to 7.9  Grades 9 - 10: level 4
8.0 to 8.9  Grades 11 - 12: level 5
9.0 to 9.9  Grades 13 - 15 (College): level 6
10 and Above    Grades 16 and Above (College Graduate): level 6

The New Dale-Chall Readability Formula:
Raw Score = 0.1579 * (PDW) + 0.0496 * ASL

"""
from __future__ import division

import random
import zho_kanjiinfo as zkj
from level import LevelDetect
from wordfilter import contain_number, is_stopword, is_latin
import codecs


KEY_AVG_SENTENCE_LENGTH = 'average_sentence_length'

class NDCLevel(LevelDetect):
    vowels = {"eng":"aeiouy",
              "deu":"aeiouyäöü",
              "fra":"ueoaiyéèêàâîùû",
              "vie":"ueoaiâăàáạãấầậẫặẵằắèéẹẽêếềễệoòóọõơờớỡợùúũụưừứữựìíịĩ",
              "jpn":"あいうえおアイウエオ"}

    def __init__(self, langid, vowels=None):
        self.langid = langid
        if langid == 'eng':
            self.default_avg_sentence_length = 10

        else: # default
            self.default_avg_sentence_length = 10

    def detect_level(self, words, extras=dict()):
        """
        Parameters:
        ---
        wordlist: a list of (string) words
        extras: dict, contains key KEY_AVG_SENTENCE_LEGNTH
        """
        if KEY_AVG_SENTENCE_LENGTH in extras:
            avgSentLength = extras[KEY_AVG_SENTENCE_LENGTH]
        else:
            avgSentLength = self.default_avg_sentence_length

        if len(words) == 0: return 0
        rawScore = 0.0
        numDifWords = 0

        #Count PDW: Percentage of Difficult words
        for word in words:
            if contain_number(word): continue
            if is_stopword(word): continue
            if self.langid=='zho':
                if is_latin(word): continue #Ignore latin words
            if self.get_singleword_level(word) >= 4:
                print(word + " is difficult ")
                numDifWords += 1

        #print(numDifWords)
        #print(len(words))
        perDifWords = (numDifWords/len(words)) * 100
        #print(perDifWords)
        rawScore = 0.1579 * (perDifWords) + 0.0496 * avgSentLength
        #print(rawScore)
        
        if perDifWords > 10:
            return self.get_map_level(rawScore + 3.6365)
        else:
            return self.get_map_level(rawScore)

    def get_map_level(self, readabilityLevel):
        """
        Map the New Dale-Chall Readability level to our pre-defined level (0-5)

        Parameters:
        ----
        readabilityLevel: New Dale-Chall Readability level (0-10+)

        Return:
        ----
        readability level (0 - 5)
        """
        d = {}
        if self.langid == 'zho':
            d = {0:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:4, 9:5, 10:6}
        else:
            d = {0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:3, 7:4, 8:4, 9:4, 10:4}
        return d.get(int(readabilityLevel), 4) #default level 5 if grade>10

    def get_singleword_level(self, word):
        if self.langid == 'zho':
            return self.convert_stroke_to_level(word)
        else:
            nsyl = self.get_number_syllables(word)
            #print(nsyl)
            if nsyl<=1:
                return 1
            elif nsyl==2:
                return 2
            elif nsyl>=3:
                return 4

    def convert_stroke_to_level(self, word):
        """
        Convert from number of strokes to the corresponding #syllables in latin
        Used for zho, jpn

        Statistics:
        Total number of strokes: 32002
        Total number of words: 3071
        In average: 10.4 strokes/word
        <10: easy
        >10: difficult
        #stroke : #words
        1: 2
        2: 16
        3: 42
        4: 100
        5: 128
        6: 189
        7: 227
        8: 289
        9: 303
        10: 326
        11: 280
        12: 306
        13: 217
        14: 201
        15: 152
        16: 100
        17: 70
        18: 38
        19: 36
        20: 16
        21: 15
        22: 6
        23: 5
        24: 2

        <4: 0
        <7: 1
        <10: 2
        <13: 3
        <16: 4
        <20: 5

        Parameters:
        ----
        word: a given word

        Return:
        ----
        number of syllables (3+ is considered to be a difficult word)
        """
        nstroke = 25 #unknown word
        try:
            totalstroke = 0
            for c in word:
                totalstroke += int(zkj.kanjimap[c][0])
            nstroke = totalstroke/len(word)
        except Exception:
            pass

        if nstroke<3: return 1
        elif nstroke<6: return 1
        elif nstroke<9: return 2
        elif nstroke<13: return 3
        elif nstroke<16: return 4
        else: return 4

    def get_number_syllables(self, word):
        """
        Count the number of syllables of a word

        Parameters:
        ----
        word: a given word

        Return:
        ----
        number of syllables (3+ is considered to be a difficult word)

        @TODO: For languages like Chinese, Korean and Japanese, this has to be adjusted
        e.g., with #strokes in Chinese
        """
        vowels = []
        try:
            vowels = self.vowels[self.langid]
        except KeyError:
            vowels = self.vowels["eng"] #default eng vowels for latin languages
        numVowels = 0
        lastWasVowel = False
        for wc in word:
            foundVowel = False
            for v in vowels:
                if v == wc:
                    if not lastWasVowel:
                        numVowels += 1   #don't count diphthongs
                    foundVowel = lastWasVowel = True
                    break
            if not foundVowel:
                lastWasVowel = False
        if len(word) > 2 and word[-2:] == "es": numVowels -= 1
        elif len(word) > 1 and word[-1:] == "e": numVowels -= 1
        return numVowels

class FNDCLevel(NDCLevel):
    def __init__(self, langid, freqDict=None, vowels=None, fc_difficulty_level=9):
        super().__init__(langid, vowels)
        self.cacheFreqDict = freqDict

        if langid == 'eng':
            self.difficulty_level = fc_difficulty_level
            self.default_avg_sentence_length = 10

        else: # default
            self.difficulty_level = fc_difficulty_level
            self.default_avg_sentence_length = 10

    def get_singleword_level(self, word, freqDict=None):
        """
        Based on the following statistics, get the corresponding level of a word
        level0 (0-250): min (0), max(7), mean(5.96)
        level 1 (250-750): min (7), max(9), mean(8.226)
        level 2 (750-1250): min (9), max(10), mean(9.02)
        level 3 (1250-2500): min (10), max(11), mean(10.2552)
        level 4 (2500-4000): min (11), max(12), mean(11.315333333333333)
        level 5 (4000-5000): min (12), max(12), mean(12.0),
        """
        if freqDict != None and word in freqDict:
            return(self.convert_fc_to_level(freqDict[word]))
        else: return super().get_singleword_level(word)


    def convert_fc_to_level(self,fc):
        d = {7:1, 8:1, 9:2, 10:3, 11:4} #<=7: 0, 12: 4 or 5
        if fc<7:
            return 0
        elif fc>=12:
            return 4
        else:
            return d[fc]

    def detect_level(self, words, extras=dict()):
        """
        Calculate readability level of a list of words
        based on counting the number of syllables of the given words

        Parameters:
        ----
        words: a list of words
        avgSentLength: average sentence length

        Return:
        ----
        readability level (from 0 - 5)

        """
        if KEY_AVG_SENTENCE_LENGTH in extras:
            avgSentLength = extras[KEY_AVG_SENTENCE_LENGTH]
        else:
            avgSentLength = self.default_avg_sentence_length

        if len(words)==0: return 0
        rawScore = 0.0
        numDifWords = 0

        #Count PDW: Percentage of Difficult words
        for word in words:
            if contain_number(word): continue
            level = self.get_singleword_level(word, self.cacheFreqDict)
            is_difficult = level >= 4
            if is_difficult: numDifWords += 1

        perDifWords = (numDifWords/len(words))*100
        rawScore = 0.1579 * (perDifWords) + 0.0496 * avgSentLength

        if perDifWords > 10:
            return self.get_map_level(rawScore + 3.6365)
        else:
            return self.get_map_level(rawScore)


def read_freqdict(filename):
    f = codecs.open(filename,'r',encoding='utf-8')
    freqdict = {}
    f.readline() #ignore first line
    for line in f:
        lines = line.strip().split('\t')
        freqdict[lines[1]] = int(lines[2])
        
    f.close()
    return freqdict

if __name__=='__main__':
    print("reading frequency dictionary..")
    freqdict = read_freqdict('../store/eng-word-byfreq.txt')
    
    
    test = ['The', 'battle', 'to', 'contain', 'the', 'wildfires', 'in', 'west-central',
    'Canada', 'has', 'reached', 'a', 'turning', 'point,', 'partly', 'thanks', 'to',
    'drizzle', 'and', 'favourable', 'winds', 'say', 'officials', 'One', 'minister', 'warned',
    'much', 'work', 'lay', 'ahead', 'but', 'we', 'may', 'be', 'turning', 'a', 'corner', 'A',
     'fifth', 'of', 'homes', 'in', 'the', 'oil', 'sands', 'city', 'of', 'Fort', 'McMurray',
     'have', 'been', 'destroyed', 'and', 'more', 'than', '80,000', 'people', 'evacuated',
     'But', 'the', 'fire', 'had', 'not', 'spread', 'as', 'fast', 'as', 'had', 'been', 'feared,',
     'said', 'Alberta', 'Premier', 'Rachel', 'Notley,', 'who', 'will', 'survey', 'the',
     'devastation', 'on', 'Monday']
    test2 = ['Last', 'Sunday', 'they', 'went', 'to', 'the', 'football', 'game']

    ndcDetect = NDCLevel('eng')
    level1 = ndcDetect.detect_level(test, {KEY_AVG_SENTENCE_LENGTH:10})
    print(level1)
    """
    fndcDetect = FNDCLevel('eng')
    level2 = fndcDetect.detect_level(test, {KEY_AVG_SENTENCE_LENGTH:27})
    print (level1, level2)
    print(fndcDetect.get_singleword_level('today'))
    print(fndcDetect.get_singleword_level('decompose'))
    print(fndcDetect.get_singleword_level('association'))
    print(fndcDetect.get_singleword_level('distribution'))
    print(fndcDetect.get_singleword_level('complicated'))
    """
    ndcDetect = NDCLevel('zho')
    print(ndcDetect.get_singleword_level('现'))
    print(ndcDetect.get_singleword_level('去'))
