#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import re, pdb
import unicodedata as ud
import sys


PUNC = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~》《”？：。，、！“…；“、—…（）‘’'

ROMANJI_MATCHER = '[a-zA-Z0-9]*'

latin_letters= {}
def is_latin(uchr):
    try: return latin_letters[uchr]
    except KeyError:
        try:
            return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))
        except Exception:
            print('unknown character ' + uchr)

def only_latin_chars(unistr):
    return all(is_latin(uchr)
           for uchr in unistr) 

def contain_no_latin(unistr):
    for uchr in unistr:
        if is_latin(uchr):
            return False
    return True

def contain_number(word):
    if re.search('[0-9]', word) != None:
        return True
    else: return False

def is_tooshort(word,minLength=2):
    if len(word)<minLength:
        return True
    else:
        return False
    
def contain_capital_middle(word):
    if is_tooshort(word):
        return False
    for i in range(1,len(word)):
        if word[i].isupper():
            return True
    return False
        
def contain_only_alpha(word):
    for uchr in word:
        if not uchr.isalpha():
            return False
    return True
    
    
def is_stopword(word):
    """
    punctuations
    """
    for p in PUNC:
        if p in word:
            return True
    return False
    
def is_only_one_char(word):
    """
    This is applied to Japanese only
    It should not contain only one character, e.g., Hiragana and Katakana
    One character kanji is accepted
    """
    HIRAGANA_ALPHABET = "あいうえお" + \
                        "かきくけこ" + \
                        "さしすせそ" + \
                        "たちつてと" + \
                        "なにぬねの" + \
                        "はひふへほ" + \
                        "まみむめも" + \
                        "らりるれろ" + \
                        "わを" + \
                        "がぎぐげご" + \
                        "ざじずぜぞ" + \
                        "だでど" + \
                        "ばびぶべぼ" + \
                        "ぱぴぷぺぽ" + \
                        "ゆやゃゅょんっよづ"

    KATAKANA_ALPHABET = "アイウエオ" + \
                        "カキクケコ" + \
                        "サシスセソ" + \
                        "タチツテト" + \
                        "ナニヌネノ" + \
                        "ハヒフヘホ" + \
                        "マミムメモ" + \
                        "ヤユヨ" + \
                        "ラリルレロ" + \
                        "ワヰヱヲ" + \
                        "ガギグゲゴ" + \
                        "ザジズゼゾ" + \
                        "ダヂヅデド" + \
                        "バビブベボ" + \
                        "パピプペポ" + \
                        "ンーュョィッェャォ"
    if len(word)==1 and (word in HIRAGANA_ALPHABET or word in KATAKANA_ALPHABET): 
        return True
    return False
    
