import numpy as np
from libc.math cimport pow as cpow
from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from adeft.nlp import stopwords
from adeft.score.permutations cimport permuter, make_permuter, \
    free_permuter, update_permuter


cdef class LongformScorer:
    cdef:
        public str shortform
        public list penalties
        public double alpha, beta, gamma, delta, rho, inv_penalty
        public dict word_scores
        int len_shortform
        dict char_map
        opt_shortform *shortform_c
        opt_params *params_c
    def __init__(self, shortform, penalties=None, alpha=0.5, beta=0.55,
                 gamma=0.4, delta=0.9, rho=0.6, inv_penalty=0.9,
                 word_scores=None):
        self.shortform = shortform.lower()
        self.len_shortform = len(shortform)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.inv_penalty = inv_penalty
        self.rho = rho
        self.params_c = make_opt_params(beta, rho)
        # Encode shortform chars as integers and build map for encoding
        # longform candidates
        self.char_map = {}
        cdef list encoded_shortform = []
        cdef int i = 0, j = 0
        for i in range(self.len_shortform):
            if self.shortform[i] not in self.char_map:
                self.char_map[self.shortform[i]] = j
                j += 1
            encoded_shortform.append(self.char_map[self.shortform[i]])
        if penalties is not None:
            self.penalties = penalties
        else:
            self.penalties = [gamma*delta**i
                              for i in range(self.len_shortform)]
        self.shortform_c = create_shortform(encoded_shortform,
                                            self.penalties)
        self.params_c = make_opt_params(beta, rho)
        if word_scores is None:
            self.word_scores = {word: 0.2 for word in stopwords}
        else:
            self.word_scores = word_scores

    cdef double get_word_score(self, str token):
        if token in self.word_scores:
            return self.word_scores[token]
        else:
            return 1.0

    cdef candidates_array *process_candidates(self, list candidates):
        cdef:
            double word_score
            str token
            list encoded_candidates, coded, prizes,
            list token_prizes, word_prizes, W
            int n, i, j
        encoded_candidates = []
        prizes = []
        word_prizes = []
        n = len(candidates)
        for i in range(n):
            coded = []
            token_prizes = []
            token =  candidates[i].lower()
            m = len(token)
            for j in range(m):
                if token[j] in self.char_map:
                    coded.append(self.char_map[token[j]])
                    token_prizes.append(self.alpha**j)
            if coded:
                encoded_candidates.append(coded)
                prizes.append(token_prizes)
                word_score = self.get_word_score(token)
                word_prizes.append(word_score)
        W = [word_prizes[-1]]
        for i in range(1, len(word_prizes)):
            W.append(W[i-1] + word_prizes[-i])
        return make_candidates_array(encoded_candidates,
                                     prizes, word_prizes, W)

    cdef tuple get_score_results(self,
                                 list candidates,
                                 list scores,
                                 double *W):
        """Produce useable output for longform scorer
        """
        n = len(candidates)
        i = j = 0
        shortform_chars = set(self.shortform)
        results = []
        best_score = -1.0
        current_score = 0.0
        current_candidates_list = []
        best_candidate = ''
        while i < n:
            current_candidates_list.append(candidates[n-i-1])
            current_candidate = ' '.join(current_candidates_list[::-1])
            if set(candidates[n-i-1]) & shortform_chars:
                current_score = scores[j]
                j += 1
            else:
                current_score *= (W[i]/(W[i] + 1))**(1 - self.rho)
            results += (current_candidate, current_score)
            if current_score > best_score:
                best_score = current_score
                best_candidate = current_candidate
            i += 1
        return (best_candidate, best_score, results)
            
    def score(self, candidates):
        cdef:
            int i, j, k, n
            double current_score, best_score, w, W, ub_char_scores
            double ub_word_scores
            double_array *best_char_scores
            double_array *previous_word_scores
            list scores
            candidates_array *candidates_c
            opt_results *results
            opt_results *probe_results
        results = make_opt_results(self.len_shortform)
        probe_results = make_opt_results(self.len_shortform)
        candidates_c = self.process_candidates(candidates)
        n = candidates_c.length
        scores = [None]*n
        best_score = -1.0
        best_char_scores = make_double_array(self.len_shortform)
        for i in range(self.len_shortform):
            best_char_scores.array[i] = -1e20
        for i in range(1, n + 1):
            ub_char_scores = 0.0
            probe(candidates_c.array[n - i], candidates_c.prizes[n - i],
                  self.shortform_c.y, self.params_c.beta,
                  probe_results)
            for j in range(self.len_shortform):
                if probe_results.char_scores[j] > best_char_scores.array[j]:
                    ub_char_scores += probe_results.char_scores[j]
                else:
                    ub_char_scores += best_char_scores.array[j]
            if i > 1:
                previous_word_scores = make_double_array(i)
                for k in range(i):
                    previous_word_scores.array[k] = \
                        candidates_c.word_prizes[n-k-1]
                ub_word_scores = opt_selection(previous_word_scores,
                                               self.len_shortform-1)
                free_double_array(previous_word_scores)
            else:
                ub_word_scores = 0.0
            ub_char_scores = ub_char_scores/self.len_shortform
            W = candidates_c.W_array[i-1]
            w = candidates_c.word_prizes[n-i]
            ub_word_scores = (w + ub_word_scores)/W
            upper_bound = (cpow(ub_char_scores, self.params_c.rho) *
                           cpow(ub_word_scores, 1-self.params_c.rho))
            if upper_bound < best_score:
                scores[i-1] = scores[i-2]*(W - w)/W
                continue
            perm_search(candidates_c, self.shortform_c,
                        self.params_c,
                        self.inv_penalty, i, results)
            current_score = results.score
            scores[i-1] = results.score
            if current_score >= best_score:
                best_score = current_score
                for j in range(self.len_shortform):
                    best_char_scores.array[j] = results.char_scores[j]
        
        free_double_array(best_char_scores)
        free_opt_results(probe_results)
        free_opt_results(results)
        free_candidates_array(candidates_c)
        return self.get_score_results(candidates, scores,
                                      candidates_c.W_array)


cdef double opt_selection(double_array *word_prizes, int k):
    cdef:
        int min_index, i, j
        double min_value, temp, output
    if k >= word_prizes.length:
        output = 0.0
        for i in range(word_prizes.length):
            output += word_prizes.array[i]
        return output
    for i in range(k):
        min_index = i
        min_value = word_prizes.array[i]
        for j in range(i+i, word_prizes.length):
            if word_prizes.array[j] < min_value:
                min_index = j
                min_value = word_prizes.array[j]
                temp = word_prizes.array[min_index]
                word_prizes.array[min_index] = word_prizes.array[i]
                word_prizes.array[i] = temp
    output = 0.0
    for i in range(k):
        output += word_prizes.array[i]
    return output
                


cdef opt_results *make_opt_results(int len_y):
    cdef opt_results *results
    results = <opt_results *> PyMem_Malloc(sizeof(opt_results))
    results.char_scores = <double *> PyMem_Malloc(len_y * sizeof(double))
    return results


cdef void free_opt_results(opt_results *results):
    PyMem_Free(results.char_scores)
    PyMem_Free(results)
    return


cdef int_array *make_int_array(int length):
    cdef int_array *output
    output = <int_array *> PyMem_Malloc(sizeof(int_array))
    output.array = <int *> PyMem_Malloc(length * sizeof(int))
    output.length = length
    return output


cdef void free_int_array(int_array *x):
    PyMem_Free(x.array)
    PyMem_Free(x)
    return


cdef double_array *make_double_array(int length):
    cdef double_array *output
    output = <double_array *> PyMem_Malloc(sizeof(int_array))
    output.array = <double *> PyMem_Malloc(length * sizeof(double))
    output.length = length
    return output


cdef void free_double_array(double_array *x):
    PyMem_Free(x.array)
    PyMem_Free(x)
    return


@boundscheck(False)
@wraparound(False)
cdef candidates_array *make_candidates_array(list encoded_candidates,
                                             list prizes,
                                             list word_prizes,
                                             list W):
    cdef:
        int i, j, num_candidates, m1,m2, n, cum_length
        candidates_array *candidates
    n = len(encoded_candidates)
    candidates = <candidates_array *> PyMem_Malloc(sizeof(candidates_array))
    candidates.array = <int_array **> PyMem_Malloc(n * sizeof(int_array*))
    candidates.prizes = <double_array **> \
        PyMem_Malloc(n * sizeof(double_array*))
    candidates.word_prizes = <double *> PyMem_Malloc(n * sizeof(double))
    candidates.cum_lengths = <int *> PyMem_Malloc(n * sizeof(int))
    candidates.W_array = <double *> PyMem_Malloc(n * sizeof(double))
    candidates.length = n
    cum_length = 0
    for i in range(n):
        m1 = len(encoded_candidates[i])
        m2 = len(encoded_candidates[n-i-1])
        candidates.array[i] = make_int_array(m1)
        candidates.prizes[i] = make_double_array(m1)
        cum_length += m2
        candidates.cum_lengths[i] = cum_length
        candidates.word_prizes[i] = word_prizes[i]
        candidates.W_array[i] = W[i]
        for j in range(m1):
            candidates.array[i].array[j] = encoded_candidates[i][j]
            candidates.prizes[i].array[j] = prizes[i][j]
    return candidates


cdef void free_candidates_array(candidates_array *candidates):
    cdef:
        int i, j
    for i in range(candidates.length):
         free_int_array(candidates.array[i])
         free_double_array(candidates.prizes[i])
    PyMem_Free(candidates.word_prizes)
    PyMem_Free(candidates.W_array)
    PyMem_Free(candidates.prizes)
    PyMem_Free(candidates.array)
    PyMem_Free(candidates.cum_lengths)
    PyMem_Free(candidates)


cdef opt_input *make_opt_input(int n, int num_words):
    cdef opt_input *input_
    input_ = <opt_input *> PyMem_Malloc(sizeof(opt_input))
    input_.x = make_int_array(n)
    input_.prizes = make_double_array(n)
    input_.word_boundaries = <unsigned int *> \
        PyMem_Malloc(num_words * sizeof(unsigned int))
    input_.word_prizes = make_double_array(num_words)
    return input_


cdef void free_opt_input(opt_input *input_):
    free_int_array(input_.x)
    free_double_array(input_.prizes)
    PyMem_Free(input_.word_boundaries)
    PyMem_Free(input_.word_prizes)
    PyMem_Free(input_)


cdef opt_params *make_opt_params(double beta, double rho):
    cdef opt_params *params
    params = <opt_params *> PyMem_Malloc(sizeof(opt_params))
    params.beta = beta
    params.rho = rho
    return params


cdef void free_opt_params(opt_params *params):
    PyMem_Free(params)


cdef opt_shortform *make_opt_shortform(int m):
    cdef opt_shortform *shortform
    shortform = <opt_shortform *> PyMem_Malloc(sizeof(opt_shortform))
    shortform.y = make_int_array(m)
    shortform.penalties = make_double_array(m)
    return shortform


cdef opt_shortform *create_shortform(list encoded_shortform,
                                     list penalties):
    cdef:
        opt_shortform *shortform
        int i = 0
        int m = len(encoded_shortform)
    shortform = make_opt_shortform(m)
    for i in range(m):
        shortform.y.array[i] = encoded_shortform[i]
        shortform.penalties.array[i] = penalties[i]
    return shortform


cdef void free_opt_shortform(opt_shortform *shortform):
    free_int_array(shortform.y)
    free_double_array(shortform.penalties)
    PyMem_Free(shortform)


@wraparound(False)
cdef void perm_search(candidates_array *candidates,
                      opt_shortform *shortform,
                      opt_params *params,
                      float inv_penalty,
                      int n,
                      opt_results *output):
    cdef:
        int input_size, i
        double current_score
        permuter *perms
        opt_input *current
        opt_results *results

    results = make_opt_results(shortform.y.length)
    total_length = candidates.cum_lengths[n - 1]
    if shortform.y.length > total_length + 1:
        input_size = total_length + shortform.y.length
    else:
        input_size = 2*total_length + 1
    current = make_opt_input(input_size, n)
    perms = make_permuter(n)
    stitch(candidates, perms.P, n, current)
    optimize(current, shortform, params, results)
    output.score = results.score
    for i in range(shortform.y.length):
        output.char_scores[i] = results.char_scores[i]
    while perms.m != 0:
        update_permuter(perms)
        stitch(candidates, perms.P, n, current)
        perm = [perms.P[i] for i in range(n)]
        optimize(current, shortform, params, results)
        current_score = results.score * cpow(inv_penalty,
                                             perms.inversions)
        if current_score > output.score:
            output.score = current_score
            for i in range(shortform.y.length):
                output.char_scores[i] = results.char_scores[i]
    free_permuter(perms)
    free_opt_results(results)
    free_opt_input(current)


@boundscheck(False)
@wraparound(False)
cdef void probe(int_array *next_token, double_array *char_scores,
                int_array *y, double beta,
                opt_results *probe_results):
    cdef:
        int i, j, input_size
        opt_input *input_
        opt_results *result
        opt_params *params
        opt_shortform *shortform
    # First initalize the probe
    if y.length > next_token.length + 1:
        input_size = next_token.length + y.length
    else:
        input_size = 2*next_token.length + 1
    input_ = make_opt_input(input_size, 1)
    result = make_opt_results(y.length)
    params = make_opt_params(beta, 1.0)
    shortform = make_opt_shortform(y.length)
    for i in range(y.length):
        shortform.y.array[i] = y.array[i]
        shortform.penalties.array[i] = 0.0
    input_.x.array[0] = -1
    input_.prizes.array[0] = 0
    i = 0
    j = 1
    for i in range(next_token.length):
        input_.x.array[j] = next_token.array[i]
        input_.x.array[j+1] = -1
        input_.prizes.array[j] = char_scores.array[i]
        input_.prizes.array[j+1] = 0
        j += 2
    while j < input_.x.length:
        input_.x.array[j] = -1
        input_.prizes.array[0] = 0
        j += 1
    input_.word_prizes.array[0] = 0
    input_.word_boundaries[0] = j - 1
    input_.W = 1
    optimize(input_, shortform, params, result)
    free_opt_input(input_)
    free_opt_params(params)
    free_opt_shortform(shortform)


@boundscheck(False)
@wraparound(False)
cdef void stitch(candidates_array *candidates, int *permutation,
                  int len_perm, opt_input *result):
    cdef int i, j, k, current_length, n, p
    n = candidates.length
    # stitched output begins with wildcard represented by -1
    result.x.array[0] = -1
    result.prizes.array[0] = 0
    i = 0
    j = 1
    for i in range(len_perm):
        p = permutation[i]
        current_length = candidates.array[n-len_perm+p].length
        for k in range(current_length):
            result.x.array[j] = \
                candidates.array[n-len_perm+p].array[k]
            # insert wildcard after each element from input
            result.x.array[j+1] = -1
            result.prizes.array[j] = \
                candidates.prizes[n-len_perm+p].array[k]
            result.prizes.array[j+1] = 0
            j += 2
        result.word_prizes.array[i] = candidates.word_prizes[n-len_perm+p]
        result.word_boundaries[i] = j - 1
    while j < result.x.length:
        result.x.array[j] = -1
        result.prizes.array[0] = 0
        j += 1
    result.W = candidates.W_array[len_perm - 1]


@boundscheck(False)
@wraparound(False)
cdef void optimize(opt_input *input_, opt_shortform *shortform,
                    opt_params *params, opt_results *output):
    """Subsequence match optimization algorithm for longform scoring

    Uses a dynamic programming algorithm to find optimal instance of
    y as a subsequence in x where elements of x each have a corresponding
    prize. Wildcard characters are allowed in x that match any element of y
    and penalties may be given for when an element of y matches a wildcard
    instead of a regular element of x. Results are updated in place.

    Paramters
    ---------
    x : C array of int
        Contains nonnegative long ints for supersequence in which we seek
        a subsequence match. May also contain the value -1 which corresponds to a
        wildcard that matches any nonnegative int.

    y : C array of int
        Sequence we seek an optimal subsequence match of in x

    prizes : C array of double
        Must be the same length as x. Prize gained for matching an element
        of y to the corresponding element of x
 : C array of double
        Must the the same length as y. Penalty lost if the corresponding
        element of y matches a wildcard.

    results : struct opt_results
        opt_results structure to where output is to be placed
    """
    cdef:
        unsigned int n = input_.x.length
        unsigned int m = shortform.y.length
        double possibility1, possibility2, word_score, char_score, w
        unsigned int i, j, k

    # Dynamic initialization of score_lookup array and traceback pointer
    # array
    cdef:
        double **score_lookup = (<double **>
                                 PyMem_Malloc((n+1) * sizeof(double *)))
        double **char_scores = (<double **>
                                PyMem_Malloc((n+1) * sizeof(double *)))
        double **word_scores = (<double **>
                                PyMem_Malloc((n+1) * sizeof(double *)))
        int **word_use = <int **> PyMem_Malloc((n+1) * sizeof(int *))
        int **pointers = <int **> PyMem_Malloc(n * sizeof(int *))
    for i in range(n+1):
        score_lookup[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        char_scores[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        word_scores[i] = <double *> PyMem_Malloc((m+1) * sizeof(double))
        word_use[i] = <int *> PyMem_Malloc((m+1) * sizeof(int))
        if i != n:
            pointers[i] = <int *> PyMem_Malloc(m * sizeof(int))
    # Initialize lookup array
    score_lookup[0][0] = 0
    char_scores[0][0] = 0
    word_scores[0][0] = 0
    word_use[0][0] = 0
    for j in range(1, m+1):
        score_lookup[0][j] = -1e20
        char_scores[0][j] = 0
        word_scores[0][j] = 0
        word_use[0][j] = 0
    for i in range(1, n+1):
        for j in range(0, m+1):
            score_lookup[i][j] = 0
            char_scores[i][j] = 0
            word_scores[i][j] = 0
            word_use[i][j] = 0
    # Main loop
    k = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Case where element of x in current position matches
            # element of y in current position. Algorithm considers
            # either accepting or rejecting this match
            if input_.x.array[i-1] == shortform.y.array[j-1]:
                if word_use[i-1][j-1] == 0:
                    w = input_.word_prizes.array[k]
                else:
                    w = 0
                possibility1 = score_lookup[i-1][j]
                # Calculate score if match is accepted
                char_score = char_scores[i-1][j-1]
                char_score += input_.prizes.array[i-1]/cpow(params.beta,
                                                            word_use[i-1][j-1])
                if char_score < 0.0:
                    char_score = 0.0
                word_score = word_scores[i-1][j-1] + w
                possibility2 = (cpow(char_score/m, params.rho) *
                                cpow(word_score/input_.W, (1-params.rho)))
                if (score_lookup[i-1][j-1] > -1e19 and 
                    possibility2 > possibility1):
                    score_lookup[i][j] = possibility2
                    char_scores[i][j] = char_score
                    word_scores[i][j] = word_score
                    word_use[i][j] = word_use[i-1][j-1] + 1
                    pointers[i-1][j-1] = 1
                else:
                    score_lookup[i][j] = possibility1
                    char_scores[i][j] = char_scores[i-1][j]
                    word_scores[i][j] = word_scores[i-1][j]
                    word_use[i][j] = word_use[i-1][j]
                    pointers[i-1][j-1] = 0
            # Case where element of x in current position is a wildcard.
            # May either accept or reject this match
            elif input_.x.array[i-1] == -1:
                possibility1 = score_lookup[i-1][j]
                # Score if wildcard match is accepted, skipping current
                # char in shortform
                char_score = char_scores[i-1][j-1]
                char_score -= shortform.penalties.array[j-1]
                # Take min with zero to ensure char_score doesn't become
                # negative
                if char_score < 0.0:
                    char_score = 0.0
                word_score = word_scores[i-1][j-1]
                possibility2 = (cpow(char_score/m, params.rho) *
                                cpow(word_score/input_.W, 1-params.rho))
                if (score_lookup[i-1][j-1] > -1e19 and
                    possibility2 > possibility1):
                    score_lookup[i][j] = possibility2
                    char_scores[i][j] = char_score
                    word_scores[i][j] = word_score
                    word_use[i][j] = word_use[i-1][j-1]
                    pointers[i-1][j-1] = 1
                else:
                    score_lookup[i][j] = possibility1
                    char_scores[i][j] = char_scores[i-1][j]
                    word_scores[i][j] = word_scores[i-1][j]
                    word_use[i][j] = word_use[i-1][j]
                    pointers[i-1][j-1] = 0
            # No match is possible. There is only one option to fill
            # current entry of dynamic programming lookup array.
            else:
                score_lookup[i][j] = score_lookup[i-1][j]
                char_scores[i][j] = char_scores[i-1][j]
                word_scores[i][j] = word_scores[i-1][j]
                word_use[i][j] = word_use[i-1][j]
                pointers[i-1][j-1] = 0
            # Reset word_use to zero when word_boundary is passed
            if i == input_.word_boundaries[k] + 1:
                word_use[i][j] = 0
        # Increment number of words used when boundary is passed
        if i == input_.word_boundaries[k] + 1:
            k += 1
    # Optimal score is in bottom right corner of lookup array
    score = score_lookup[n][m]
    # Free the memory used by the lookup array

    # Set score in output
    output.score = score
    # Trace backwards through pointer array to discover which elements of x
    # were matched and add the score associated to each character in the
    # shortform into the char scores array.
    i, j, k = n, m, 0
    while j > 0:
        if pointers[i - 1][j - 1]:
            i -= 1
            j -= 1
            if input_.x.array[i] == -1:
                output.char_scores[m-k-1] = -shortform.penalties.array[m-k-1]
            else:
                output.char_scores[m-k-1] = \
                    input_.prizes.array[i]/cpow(params.beta, word_use[i][j])
            k += 1
        else:
            i -= 1
    # Free lookup arrays used by algorithm
    for i in range(n+1):
        PyMem_Free(score_lookup[i])
        PyMem_Free(word_use[i])
        if i < n:
            PyMem_Free(pointers[i])
    PyMem_Free(score_lookup)
    PyMem_Free(word_use)
    PyMem_Free(pointers)

