# NOTE FOR PROJECT EVALUATOR

the files necessary to run the model are stored at folder learning_json/
DO NOT use learned_word_freq.json, use word_logp.trie instead

EVALUATE USING THE learning_json/word_logp.trie file, 
keep  use_trie = True in file

Requirements:
marisa_trie, re
