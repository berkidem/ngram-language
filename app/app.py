import streamlit as st
import numpy as np
import os
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Page Configuration
###############################################################################
st.set_page_config(
    page_title="Name Generator",
    page_icon="✍️",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Define a utility function to get the logs from the model and converts that into user-friendly format
def convert_log(log, token_to_char):
    """
    Searches for a token list in the log string and replaces it with the actual occupation names.
    """
    pattern = r"\[(.*?)\]"
    match = re.search(pattern, log)
    if match:
        token_str = match.group(0)
        try:
            # Convert the token list string to a Python list
            tokens = ast.literal_eval(token_str)
            # Convert each token to its occupation name using token_to_char mapping
            occupation_names = [
                token_to_char.get(token, str(token)) for token in tokens
            ]
            new_token_str = "[" + ", ".join(occupation_names) + "]"
            # Replace the original token list in the log with the occupation names
            new_log = log.replace(token_str, new_token_str)
            return new_log
        except Exception as e:
            # If conversion fails, return the original log
            return log


###############################################################################
# Utility: RNG and Discrete Sampler (same as in your model code)
###############################################################################
class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 in [0, 1)
        return (self.random_u32() >> 8) / 16777216.0


def sample_discrete(probs, coinf):
    """Sample from a discrete distribution `probs` given a random number `coinf` in [0,1)."""
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1  # fallback in case of floating rounding issues


###############################################################################
# Model Classes
###############################################################################
class NgramModel:
    """Exact same NgramModel as in your name-generation code snippet."""

    def __init__(self, vocab_size, seq_len, smoothing=0.0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts = np.zeros((vocab_size,) * seq_len, dtype=np.uint32)
        self.uniform = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size

    def train(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        self.counts[tuple(tape)] += 1

    def get_counts(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        return self.counts[tuple(tape)]

    def __call__(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        counts = self.counts[tuple(tape)].astype(np.float32)
        counts += self.smoothing
        s = counts.sum()
        if s > 0:
            probs = counts / s
        else:
            probs = self.uniform
        return probs


class BackoffNgramModel:
    """
    A backoff model that can handle multiple n-gram orders, just like
    the one in the occupation example. We'll store logs for debugging.
    """

    def __init__(self, vocab_size, seq_lens, smoothings, counts_threshold=0):
        """
        seq_lens: list of n-gram orders,
        smoothings: list of smoothing values for the corresponding seq_len
        """
        self.vocab_size = vocab_size
        self.counts_threshold = counts_threshold
        self.models = {}
        self.logs = []

        for seq_len, smoothing in zip(seq_lens, smoothings):
            self.models[seq_len] = NgramModel(vocab_size, seq_len, smoothing)

        # Sort from largest to smallest n-gram order for backoff
        self.seq_lens = sorted(seq_lens, reverse=True)

    def __call__(self, tape):
        """Return the probability distribution of the next token given the tape."""
        for seq_len in self.seq_lens:
            context_len = seq_len - 1
            if len(tape) >= context_len:
                context = tape[-context_len:] if context_len > 0 else []
                raw_counts = self.models[seq_len].get_counts(context)
                if raw_counts.sum() > self.counts_threshold:
                    self.logs.append(
                        f"Using {seq_len}-gram model with context {context} (counts sum: {raw_counts.sum()})"
                    )
                    return self.models[seq_len](context)

        # Fallback to the unigram model
        self.logs.append("Backed off to unigram model due to insufficient data.")
        return self.models[1]([])

    def load_counts(self, counts_dict):
        """Load counts for each n-gram order from a dict {order: np_array_of_counts}."""
        for seq_len, arr in counts_dict.items():
            if seq_len in self.models:
                if arr.shape == self.models[seq_len].counts.shape:
                    self.models[seq_len].counts = arr
                else:
                    raise ValueError(
                        f"Counts shape mismatch for {seq_len}-gram: expected {self.models[seq_len].counts.shape}, got {arr.shape}."
                    )


###############################################################################
# Plotting Utilities (same style as your occupation-based Streamlit)
###############################################################################


def display_char(token, token_to_char):
    """
    Return the user-friendly label for a given token.
    Here we rename the newline character to <EOT>.
    """
    ch = token_to_char[token]
    return "<EOT>" if ch == "\n" else ch


def plot_full_distribution(values, token_to_char, chosen_token, title):
    """
    Plot a bar chart of ALL characters in alphabetical order,
    highlighting the chosen token in red.
    """

    n = len(values)
    tokens = list(range(n))

    # Sort tokens by the display name of each character
    tokens.sort(key=lambda t: display_char(t, token_to_char))

    # Build lists of labels and values in that sorted order
    sorted_labels = [display_char(t, token_to_char) for t in tokens]
    sorted_values = [values[t] for t in tokens]

    # Prepare color coding for the chosen token
    chosen_label = display_char(chosen_token, token_to_char)
    colors = ["red" if label == chosen_label else "blue" for label in sorted_labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(n), sorted_values, color=colors)

    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_labels, rotation=60, ha="right")

    fig.tight_layout()
    return fig


def plot_sampling_intervals(prob_values, chosen_token, coinf, token_to_char, title):
    """
    Plot the entire [0,1] probability interval in alphabetical order for ALL characters.
    Highlight the chosen token's segment in red. We rename \\n to <EOT> via display_char.
    """

    n = len(prob_values)
    tokens = list(range(n))

    # Sort tokens by display name
    tokens.sort(key=lambda t: display_char(t, token_to_char))
    sorted_probs = [prob_values[t] for t in tokens]
    sorted_labels = [display_char(t, token_to_char) for t in tokens]

    fig, ax = plt.subplots(figsize=(10, 3))
    left_edge = 0.0

    for label, p in zip(sorted_labels, sorted_probs):
        right_edge = left_edge + p
        # Color the segment red if it contains the coinf
        color = "red" if left_edge <= coinf < right_edge else "blue"
        ax.barh(
            y=0,
            width=p,
            left=left_edge,
            height=0.4,
            color=color,
            edgecolor="black",
            linewidth=1,
            alpha=0.7,
        )

        # Put a label in the middle of each segment
        midpoint = (left_edge + right_edge) / 2
        ax.text(x=midpoint, y=0.45, s=label, ha="center", va="bottom", fontsize=8)

        left_edge = right_edge

    # Draw the random number as a dashed vertical line
    ax.axvline(coinf, color="black", linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Probability Interval")
    ax.set_title(title)

    fig.tight_layout()
    return fig


###############################################################################
# Load Data
###############################################################################
@st.cache_resource
def load_data():
    import pickle

    # Load the exact same vocab info from the pickle
    with open("../dev/name_vocab.pkl", "rb") as f:
        vocab_dict = pickle.load(f)
    char_to_token = vocab_dict["char_to_token"]
    token_to_char = vocab_dict["token_to_char"]
    EOT_TOKEN = vocab_dict["EOT_TOKEN"]
    vocab_size = vocab_dict["vocab_size"]

    # Now load the n-gram raw counts
    counts_dict = {}
    for seq_len in [5, 4, 3, 2, 1]:
        path = f"../dev/ngram_raw_counts_{seq_len}_gram.npy"
        arr = np.load(path)
        counts_dict[seq_len] = arr

    return char_to_token, token_to_char, EOT_TOKEN, counts_dict, vocab_size


char_to_token, token_to_char, EOT_TOKEN, counts_dict, vocab_size = load_data()


###############################################################################
# Streamlit Page Title & Intro
###############################################################################
st.title("Name Generator (Character-Level N-Gram Model)")
st.write(
    """
    This app demonstrates a simple character-level n-gram model trained on names (or any text).
    We back off from high-order n-gram to lower orders if there's insufficient data for the context.

    Enter an initial text prompt (possibly empty) and click **Generate** to keep
    sampling characters until we hit the end-of-text token (\\n). You can adjust
    smoothing, minimum counts threshold, and random seed for different variations.
    """
)

###############################################################################
# Sidebar Controls
###############################################################################
st.sidebar.header("Model Hyperparameters")

counts_threshold = st.sidebar.slider("Minimum counts threshold for backoff", 0, 200, 5)

# Smoothing for 5-,4-,3-,2-,1-grams
smoothing_5gram = st.sidebar.slider("5-gram smoothing", 0.01, 1000.0, 0.3, step=0.01)
smoothing_4gram = st.sidebar.slider("4-gram smoothing", 0.01, 1000.0, 0.3, step=0.01)
smoothing_3gram = st.sidebar.slider("3-gram smoothing", 0.01, 1000.0, 0.3, step=0.01)
smoothing_2gram = st.sidebar.slider("2-gram smoothing", 0.01, 1000.0, 0.3, step=0.01)
smoothing_1gram = st.sidebar.slider("1-gram smoothing", 0.01, 1000.0, 0.3, step=0.01)

random_seed = st.sidebar.number_input(
    "Random Seed", min_value=0, max_value=999999, value=42
)
visualize_probs = st.sidebar.checkbox("Show Probability Visuals", value=False)

print_logs = st.sidebar.checkbox("Print Inference Logs", value=False)

st.sidebar.header("Generation Control")
max_new_chars = st.sidebar.number_input(
    "Maximum characters to generate (failsafe)", min_value=1, max_value=500, value=20
)

###############################################################################
# Main User Input
###############################################################################
initial_text = st.text_input(
    "Enter some initial characters (prompt). If empty, generation starts from EOT token."
)

if st.button("Generate"):
    # Create the backoff model
    backoff_model = BackoffNgramModel(
        vocab_size=vocab_size,
        seq_lens=[5, 4, 3, 2, 1],
        smoothings=[
            smoothing_5gram,
            smoothing_4gram,
            smoothing_3gram,
            smoothing_2gram,
            smoothing_1gram,
        ],
        counts_threshold=counts_threshold,
    )
    backoff_model.load_counts(counts_dict)
    sample_rng = RNG(random_seed)

    # Convert user input to tokens. If char is not in vocab, skip it or handle gracefully
    tape = []
    for ch in initial_text:
        if ch in char_to_token:
            tape.append(char_to_token[ch])
        else:
            st.warning(f"Skipping unknown character: {repr(ch)}")

    # If no initial text, start with EOT token (like in your model's code)
    if len(tape) == 0:
        tape = [EOT_TOKEN]  # e.g. start from newline as a prompt

    generated_tokens = []
    st.header("Step-by-Step Generation")

    step = 0
    while True:
        step += 1
        if step > max_new_chars:
            st.write(f"Hit maximum of {max_new_chars} characters. Stopping.")
            break

        st.subheader(f"Step {step}")

        # Probability distribution from the model
        probs = backoff_model(tape)

        # Raw counts (before smoothing) from whichever model was used
        # The logs in backoff_model tell us the n-gram order used
        # but we must figure out that order to display raw counts.
        # We'll do it again: check from largest to smallest.
        chosen_order = None
        chosen_context = None
        for n in sorted(backoff_model.models.keys(), reverse=True):
            context_len = n - 1
            if len(tape) >= context_len:
                context = tape[-context_len:]
                arr = backoff_model.models[n].get_counts(context)
                if arr.sum() > counts_threshold:
                    chosen_order = n
                    chosen_context = context
                    break
        if chosen_order is None:
            # fallback is 1-gram
            chosen_order = 1
            chosen_context = []

        raw_counts = (
            backoff_model.models[chosen_order]
            .get_counts(chosen_context)
            .astype(np.float32)
        )
        raw_sum = raw_counts.sum()
        raw_probs = (
            raw_counts / raw_sum
            if raw_sum > 0
            else backoff_model.models[chosen_order].uniform
        )

        # Also compute smoothed probabilities the same way that model does internally
        smoothed_counts = raw_counts + backoff_model.models[chosen_order].smoothing
        smooth_sum = smoothed_counts.sum()
        if smooth_sum > 0:
            smoothed_probs = smoothed_counts / smooth_sum
        else:
            smoothed_probs = backoff_model.models[chosen_order].uniform

        # Sample a token
        coinf = sample_rng.random()
        chosen_token = sample_discrete(smoothed_probs.tolist(), coinf)

        st.markdown(
            f"**Using {chosen_order}-gram model** with context:"
            + "".join([token_to_char[token] for token in chosen_context])
        )
        chosen_char = token_to_char[chosen_token]
        st.write(
            f"Chosen character: **{repr(chosen_char)}** (random draw = {coinf:.4f})"
        )

        if visualize_probs:
            # Show bar charts for raw counts, raw probs, smoothed probs
            col1, col2, col3 = st.columns(3)
            with col1:
                fig1 = plot_full_distribution(
                    raw_counts, token_to_char, chosen_token, "Raw Counts"
                )
                st.pyplot(fig1)
            with col2:
                fig2 = plot_full_distribution(
                    raw_probs, token_to_char, chosen_token, "Raw Probabilities"
                )
                st.pyplot(fig2)
            with col3:
                fig3 = plot_full_distribution(
                    smoothed_probs,
                    token_to_char,
                    chosen_token,
                    "Smoothed Probabilities",
                )
                st.pyplot(fig3)

            # Additional interval visualization
            fig_sampling = plot_sampling_intervals(
                smoothed_probs,
                chosen_token,
                coinf,
                token_to_char,
                "Sampling from [0,1]",
            )
            st.pyplot(fig_sampling)

        # Append the chosen token
        tape.append(chosen_token)
        generated_tokens.append(chosen_token)

        # Stop if the chosen token is EOT
        if chosen_token == EOT_TOKEN:
            st.write("End-of-text token reached. Stopping generation.")
            break

    # Final Output
    st.header("Final Generated Text")
    result_str = "".join(token_to_char[t] for t in generated_tokens)
    st.write(f"**Result**: {repr(result_str)}")

    st.markdown("---")
    st.markdown("### Debug Logs")
    for log in backoff_model.logs:
        st.write(convert_log(log, token_to_char))
