import collections
import typing
from collections import Counter
from itertools import combinations

# from music21 import harmony
import math
from mirdata.datasets.beatles import *
from pydub import AudioSegment
from pydub.generators import Sine

from constants import *
from core.data.utils import split_dataset

chord_freq = Counter()


class ChordsEmbeddingDataset(Dataset):
    def __init__(
            self,
            chords_file,
            keys_file,
            k: int
    ) -> None:
        r"""
        Dataset for embedding.

        (Center chord, Context chord, Negative chords)
        Args:
            k: negative samples for one chord
        """
        super().__init__()
        normalizer = ChordsNormalizer()
        chord_and_neighbor = []
        window = 5

        for chord_file, key_file in zip(chords_file, keys_file):
            try:
                chord = load_chords(chord_file).labels
                key = load_key(key_file).keys[0]
                chord_ids = normalizer.normalize_chord_progression(key, chord)
                chord_freq.update(Counter(chord_ids))
                for gap in range(1, window + 1):
                    for chord, after in zip(chord_ids, chord_ids[gap:]):
                        chord_and_neighbor += [[chord, after]]

                    for chord, before in zip(chord_ids[gap:], chord_ids):
                        chord_and_neighbor += [[chord, before]]
            except ValueError as e:
                pass
            except FileNotFoundError as e:
                print(e)
                pass

        total_count = sum(chord_freq.values())
        chord_prob = {chord: (freq / total_count) ** 0.75 for chord, freq in chord_freq.items()}

        all_chords = np.array(list(chord_prob.keys()))
        probs = np.array(list(chord_prob.values()))
        probs /= probs.sum()  # 确保概率之和为1

        def sample_negative(k):
            return np.random.choice(all_chords, size=k, p=probs)

        for i, (center, neighbor) in enumerate(chord_and_neighbor):
            negative = sample_negative(k)
            chord_and_neighbor[i] = (center, (neighbor, negative))

        self.data = chord_and_neighbor

    @classmethod
    def with_data(
            cls,
            data
    ):
        dataset = cls([], [], 0)
        dataset.data = data
        return dataset

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> typing.Tuple[typing.Any, typing.Any]:
        return self.data[index]


class ChordsProgressionDataset(Dataset):
    def __init__(
            self,
            chords_files,
            keys_files
    ) -> None:
        r"""
        Dataset for chord progression

        chord_ids
        """
        super().__init__()
        normalizer = ChordsNormalizer()
        chord_sequences = []
        for chord_file, key_file in zip(chords_files, keys_files):
            try:
                chords = load_chords(chord_file).labels
                key = load_key(key_file).keys[0]
                chord_ids = normalizer.normalize_chord_progression(key, chords)
                chord_sequences += [chord_ids]
            except ValueError as _e:
                pass
            except FileNotFoundError as e:
                print(e)
                pass

        self.data = chord_sequences

    @classmethod
    def with_data(
            cls,
            data
    ):
        dataset = cls([], [])
        dataset.data = data
        return dataset

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> typing.Tuple[typing.Any, typing.Any]:
        return self.data[index], None


# 4 notes supported at max
max_notes_in_a_chord = 6
chord_cnts = [math.comb(12, cnt) for cnt in range(2, max_notes_in_a_chord + 1)]
TOTAL_CHORD_CNT = sum(chord_cnts)

notes_to_chord_id_map = collections.defaultdict(int)
chord_id_to_notes_list = []

numeric_note_to_int_map = {
    '1': 0,
    '2': 2,
    '3': 4,
    '4': 5,
    '5': 7,
    '6': 9,
    '7': 11,
}

alphabetic_note_to_int_map = {
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11,
}


def numeric_note_to_int(note: str):
    r"""
    '1' -> 0
    '2' -> 2
    '3' -> 4
    '4' -> 5
    'b5' -> 6
    """
    # TODO: what's this '*' ?
    note = note.replace("*", "")
    diff = 0
    diff_note = ''.join([c for c in note if not c.isnumeric()])
    note = ''.join([c for c in note if c.isnumeric()])
    if diff_note == 'b':
        diff = -1
    elif diff_note == '#':
        diff = 1
    elif diff_note:
        assert False
    # normalize, 9 -> 2
    note = str(((int(note) - 1) % 7) + 1)
    return numeric_note_to_int_map[note] + diff


def alpha_note_to_int(note: str):
    r"""
    C -> 0
    D -> 2
    E -> 4
    F -> 5
    bG -> 6
    """
    diff = 0
    if len(note) == 2:
        diff_note = note[1]
        note = note[0]
        if diff_note == 'b':
            diff = -1
        elif diff_note == '#':
            diff = 1
        else:
            assert False
    return alphabetic_note_to_int_map[note] + diff


def prepare():
    global chord_id_to_notes_list
    for notes_cnt in range(2, max_notes_in_a_chord + 1):
        for i, comb in enumerate(combinations(range(12), notes_cnt)):
            comb = list(comb)
            comb.sort()
            comb = tuple(comb)
            notes_to_chord_id_map[comb] = len(notes_to_chord_id_map)
            chord_id_to_notes_list += [comb]


prepare()


def notes_to_chord_id(chord_notes: list):
    chord_notes.sort()
    return notes_to_chord_id_map[tuple(chord_notes)]
    # prefix = sum(chord_cnts[:len(chord_notes) - 2])
    # if len(chord_notes) <= max_notes_in_a_chord:
    #     return prefix + combination_index(chord_notes, 12)
    # else:
    #     raise RuntimeError("unsupported chord")


class ChordsNormalizer:
    def __init__(self) -> None:
        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # self.note_to_int = {note: i for i, note in enumerate(self.notes)}
        self.chord_structures = {
            'maj': [0, 4, 7],
            'min': [0, 3, 7],
            'dim': [0, 3, 6],
            'aug': [0, 4, 8],
            '7': [0, 4, 7, 10],  # 属七和弦
            'dom7': [0, 4, 7, 10],  # 属七和弦
            'maj6': [0, 4, 7, 9],  # 大七和弦
            'maj7': [0, 4, 7, 11],  # 大七和弦
            'min7': [0, 3, 7, 10],  # 小七和弦
            'minmaj7': [0, 3, 7, 11],  # 小大七和弦
            'dim7': [0, 3, 6, 9],  # 减七和弦
            'hdim7': [0, 3, 6, 10],  # 半减七和弦，也称为小七减五和弦
            'min7b5': [0, 3, 6, 10],  # 小七减五和弦，与半减七和弦相同
            'aug7': [0, 4, 8, 10],  # 增七和弦
            'augmaj7': [0, 4, 8, 11],  # 增大七和弦
            'sus2': [0, 2, 7],  # 挂二和弦
            'sus4': [0, 5, 7],  # 挂四和弦
            '7sus4': [0, 5, 7, 10],  # 属七挂四和弦
            '6': [0, 4, 7, 9],  # 六和弦
            'min6': [0, 3, 7, 9],  # 小六和弦
            '9': [0, 4, 7, 10, 14],  # 九和弦
            'maj9': [0, 4, 7, 11, 14],  # 大九和弦
            'maj(9)': [0, 4, 7, 14],  # add 9
            'min9': [0, 3, 7, 10, 14],  # 小九和弦
        }
        import itertools
        intervals = collections.defaultdict(lambda: collections.defaultdict(int))
        for a, b in itertools.combinations(range(len(self.notes)), 2):
            note_a, note_b = self.notes[a], self.notes[b]
            diff = abs(a - b)
            intervals[note_a][note_b] = intervals[note_b][note_a] = diff
        self.intervals = intervals

    def normalize(self, normalized):
        pass

    def normalize_chord_progression(self, key, sequence):

        def parse_chord(chord_name: str):

            comma = int(chord_name.find(':'))
            slash = int(chord_name.find('/'))

            root = chord_name[:min(comma, slash) if min(comma, slash) > -1 else max(comma, slash)] if max(comma,
                                                                                                          slash) != -1 else chord_name
            chord_type = 'maj' if comma == -1 else (
                chord_name[comma + 1:] if slash == -1 else chord_name[comma + 1:slash])
            lowest_note = None if slash == -1 else chord_name[slash + 1:]

            # 处理根音可能包含升降号的情况
            if len(root) > 1 and (root[1] == '#' or root[1] == 'b'):
                root_note = root[:2]
                suffix = root[2:]
            else:
                root_note = root[0]
                suffix = root[1:]

            # 返回根音和和弦类型
            return root_note, suffix + chord_type, lowest_note

        def get_chord_notes(chord_type: str):
            if chord_type in self.chord_structures:
                return self.chord_structures[chord_type][:]
            if chord_type[0] == '(' and chord_type[-1] == ')':
                chord_type = chord_type[1:-1]
            else:
                left = chord_type.find("(")
                if left != -1 and chord_type[-1] == ")":
                    added = chord_type[left + 1:-1]

                    added = [numeric_note_to_int(add) for add in added.split(',')]
                    return added + get_chord_notes(chord_type[:left])
                raise RuntimeError(f"unrecognized chord type {chord_type}")
            # notes separated by ','
            notes = [0] + [numeric_note_to_int(note) for note in chord_type.split(',')]
            return notes

        def get_absolute_chord_notes(root, chord_type, lowest_note):
            r"""
            root: note name
            chord_type: maj, min, aug, dim, (1,5)
            lowest_note: also transposed chord, which will be neglected
            """
            if root == 'N':
                return []

            # 获取根音的索引
            root_index = alpha_note_to_int(root)

            # 获取和弦结构
            structure = get_chord_notes(chord_type)

            if lowest_note:
                structure += [numeric_note_to_int(lowest_note)]

            # 计算和弦中的音符
            return [(root_index + interval) % 12 for interval in structure]

        def transpose_chord(chord_notes, interval):
            return [(note + interval) % 12 for note in chord_notes]

        ids = []
        for chord in sequence:
            # 解析和弦
            root_note, chord_type, lowest_note = parse_chord(chord)

            # 获取和弦的绝对音符
            chord_notes = get_absolute_chord_notes(root_note, chord_type, lowest_note)
            chord_notes = list(set(chord_notes))
            if not chord_notes:
                continue

            interval = self.intervals[root_note][key]
            # 转置和弦到目标调性
            transposed_chord_notes = transpose_chord(chord_notes, interval)

            # 对转置后的和弦进行独热编码
            chord_id = notes_to_chord_id(transposed_chord_notes)

            ids += [chord_id]

        # 独热编码转置的和弦
        return ids


def chord_id_to_notes(id):
    return list(chord_id_to_notes_list[id])
    # def first_prefix_sum_exceeds(lst, K):
    #     return next((i for i, x in enumerate(accumulate(lst), 1) if x > K), -1) - 1
    # notes_cnt = first_prefix_sum_exceeds(chord_cnts, id) + 2
    # id = id - sum(chord_cnts[:notes_cnt - 2])
    # notes = []
    #
    # for note_id in range(notes_cnt):
    #     remain_notes_cnt = notes_cnt - note_id - 1
    #     notes += [id // math.comb(12 - note_id, remain_notes_cnt)]
    #     id -= id // math.comb(12 - note_id, remain_notes_cnt) * math.comb(12 - note_id, remain_notes_cnt)

    # return notes


def get_beatles_files():
    chords = []
    keys = []
    dir = DATASET_PATH / "beatles" / "chordlab" / "The Beatles"
    for album in os.listdir(dir):
        if os.path.isdir(dir / album):
            for song in os.listdir(dir / album):
                if song.endswith(".lab"):
                    # ext = os.path.split(song)
                    # song_name = song[:-len(ext)]
                    chords += [str(dir / album / song)]
                    keys += [str(DATASET_PATH / "beatles" / "keylab" / "The Beatles" / album / song)]
    return chords, keys


def build_chord_progression_dataset(chords, keys, ratios):
    base_dataset = ChordsProgressionDataset(chords, keys)
    return [
        ChordsProgressionDataset.with_data(dataset.dataset.data)
        for dataset in split_dataset(base_dataset, ratios, )
    ]


def build_chord_embedding_dataset(chords, keys, ratios, k):
    base_dataset = ChordsEmbeddingDataset(chords, keys, k=k)
    return [
        ChordsEmbeddingDataset.with_data(dataset)
        for dataset in split_dataset(base_dataset, ratios, )
    ]


def generate_tone(frequency, duration):
    # 生成一个正弦波音频片段
    tone = Sine(frequency)
    # 设置音频的持续时间和声道（这里是单声道）
    audio = tone.to_audio_segment(duration=duration).set_channels(1)
    return audio


def twelve_note_to_frequency(note: int, reference_pitch=261.63):
    """
    0 -> 261.63
    """

    def numeric_note_to_midi_number(note, octave=5):
        """
        C -> 60
        """
        return octave * 12 + note

    return reference_pitch * (2 ** ((numeric_note_to_midi_number(note) - 60) / 12.0))


def chord_to_audio(notes):
    duration = 1000  # 持续时间，以毫秒为单位

    # 生成和弦
    db = -1
    chord = AudioSegment.silent(duration=duration)
    for frequency in [twelve_note_to_frequency(n) for n in notes]:
        chord = chord.overlay(generate_tone(frequency, duration), gain_during_overlay=db)

    # 导出和弦音频

    # path = OUTPUT_DIR / "chord.wav"
    # chord.export(path, format="wav")
    # print(f"saved to {path}")
    return chord


def chords_to_audio(chords):
    duration = 1000  # 持续时间，以毫秒为单位

    # 生成和弦
    audio = AudioSegment.silent(duration=duration)
    for chord in chords:
        audio = audio.append(chord_to_audio(chord))

    # 导出和弦音频
    path = OUTPUT_DIR / "audio.wav"
    audio.export(path, format="wav")
    print(f"saved to {path}")


def test_reconstruct():
    id = notes_to_chord_id(chords)

    reconstructed_notes = chord_id_to_notes(id)

    assert chords == reconstructed_notes


if __name__ == "__main__":
    # major
    # I -> I7 -> IV -> IV min
    chords = [
        [0, 4, 7],
        [0, 4, 7, 10],
        [5, 9, 12],
        [5, 8, 12],
    ]
    chords_to_audio(chords)
