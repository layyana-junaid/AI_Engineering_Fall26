### Split the raw text into words
def pre_tokenize(text):
    words = []
    current = ""
    for ch in text:
        if ch == " ":
            if current:
                words.append(current)
            current = " "          
        else:
            current += ch
    if current:
        words.append(current)
    return words

#count word frequenices that we have
def count_words(words):
    freqs = {}
    for w in words:
        freqs[w] = freqs.get(w, 0) + 1
    return freqs


#count every adjacent symbol pair across the corpus
def count_pairs(word_freqs, splits):
    pair_counts = {}
    for word, freq in word_freqs.items():
        symbols = splits[word]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
    return pair_counts


#merge the most frequent pair in all words
def merge_pair(pair, splits):
    a, b = pair
    for word in splits:
        symbols = splits[word]
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        splits[word] = merged
    return splits


#greedily learn and merge the most frequent pair of symbols until we reach the target vocabulary size
def train_bpe(text, target_vocab_size, verbose=True):
    words = pre_tokenize(text)
    word_freqs = count_words(words)

    vocab = []
    for w in word_freqs:
        for ch in w:
            if ch not in vocab:
                vocab.append(ch)
    vocab.sort()

    splits = {w: [ch for ch in w] for w in word_freqs}

    merges = []  
    while len(vocab) < target_vocab_size:
        pair_counts = count_pairs(word_freqs, splits)
        if not pair_counts:
            break

        best_pair, best_count = None, 0
        for pair, count in pair_counts.items():
            if count > best_count:
                best_pair, best_count = pair, count

        if best_count < 2:   
            break

        splits = merge_pair(best_pair, splits)
        merges.append(best_pair)
        new_symbol = best_pair[0] + best_pair[1]
        vocab.append(new_symbol)
        if verbose:
            print(f"  merge #{len(merges):>2}: {best_pair[0]!r} + {best_pair[1]!r} "
                  f"-> {new_symbol!r}   (count = {best_count})")

    vocab.append("<unk>")   
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return merges, token_to_id, id_to_token


#encode a new text using the learned merges and vocabulary
def encode(text, merges, token_to_id):
    tokens = []
    for word in pre_tokenize(text):
        symbols = [ch for ch in word]
        for a, b in merges:                 
            merged = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        tokens.extend(symbols)

    ids = [token_to_id[t] if t in token_to_id else token_to_id["<unk>"]
           for t in tokens]
    return tokens, ids


#decode a list of ids back into text using the learned vocabulary
def decode(ids, id_to_token):
    return "".join(id_to_token[i] for i in ids)


if __name__ == "__main__":
    sample = ("the running dog and the jumping dog and the singing bird and the running cat that involves building a chatbot")

    print("Training:")
    print(f"  {sample!r}\n")

    print("Most frequent pair first):")
    merges, token_to_id, id_to_token = train_bpe(sample, target_vocab_size=30)

    print(f"Final Vocab ({len(token_to_id)} tokens):")
    print(" ", list(token_to_id.keys()))

    test = "the running dog"
    tokens, ids = encode(test, merges, token_to_id)
    print(f"Encoding another text: {test!r}")
    print(f"  tokens: {tokens}")
    print(f"  ids:    {ids}")

    round_trip = decode(ids, id_to_token)
    print(f"Decoding it back: {round_trip!r}")
    print(f"  lossless round-trip: {round_trip == test}")
