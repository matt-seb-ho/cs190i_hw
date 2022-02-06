def count_uni(txt):
	counts = {}
	for char in txt:
		if char not in counts:
			counts[char] = 0
		counts[char] += 1
	return counts


def count_bi(txt):
	bigrams = [(s1, s2) for s1, s2 in zip(txt, txt[1:])]
	print('bigram len:', len(bigrams))
	counts = {}
	for bigram in bigrams:
		if bigram not in counts:
			counts[bigram] = 0
		counts[bigram] += 1
	return counts

corpus = 'AAABANBABBBNNANBNN#'


# uni_counts = count_uni(corpus)
# bi_counts = count_bi(corpus)
# for key in bi_counts:
# 	bi = key[0] + key[1]
# 	print("P(`{bi}') = count(`{bi}') / count(`{first}') = {c1} / {c2} = ".format(bi=bi, first=key[0], c1=bi_counts[key], c2=uni_counts[key[0]]))

txt = 'ABANABB#'
for s1, s2 in zip(txt, txt[1:]):
	print('P(' + s2 + ' \mid ' + s1 + ')', end='')
print()

