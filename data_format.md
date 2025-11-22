### `seq.jsonl`

Each line represents a single user’s behavior sequence, ordered chronologically.

**Example entry (one user’s sequence):**

```json
[
  [481, 53968, null, {"112": 18, "117": 103, "118": 125, "119": 87, "120": 126, "100": 2, "101": 40, "102": 8559, "122": 5998, "114": 16, "116": 1, "121": 52176, "111": 5630}, 0, 1746077791],
  [481, 50652, null, {"112": 12, "117": 285, "118": 737, "119": 1500, "120": 1071, "100": 6, "101": 22, "102": 10420, "122": 2269, "114": 16, "115": 43, "116": 13, "121": 12734, "111": 6737}, 1, 1746094091],
  [481, 23145, null, {"112": 11, "117": 84, "118": 774, "119": 1668, "120": 348, "100": 6, "101": 32, "102": 6372, "122": 2980, "114": 16, "116": 15, "121": 30438, "111": 34195}, 2, 1746225104]
  ...
]
```

**Record format:**

```text
[user_id, item_id, user_feature, item_feature, action_type, timestamp]
```

* **User profile record**

  * `item_id`, `item_feature`, and `action_type` are `null`
  * `user_feature` is a dictionary containing user-level features

* **Item interaction record**

  * `user_feature` is `null`
  * `item_id`, `item_feature`, and `action_type` describe the item interaction

---

### Feature format

Both `user_feature` and `item_feature` fields are dictionaries of the form:

```json
{
  "feature_id": feature_value_reid,
  ...
}
```

Feature IDs are stored as strings; their values are re-indexed feature value IDs.

---

### ID re-indexing

* `user_id`, `item_id`, and all feature values are re-indexed into consecutive integers starting from 1 for use in embedding lookups.
* The mapping from raw IDs to re-indexed IDs is stored in `indexer.pkl`.

---

### `indexer.pkl`

A Python pickle file that stores all mappings from original IDs to re-indexed IDs:

* **`indexer['u']`** – a list of `(raw_user_id, user_reid)` pairs:

  ```python
  [
    ('user_1109670', 292),
    ('user_1091939', 364),
    ...
  ]
  ```

* **`indexer['i']`** – a list of `(raw_item_id, item_reid)` pairs:

  ```python
  [
    (30001221920, 42033),
    (30002476010, 51725),
    ...
  ]
  ```

* **`indexer['f']`** – per-feature mappings of raw feature values to re-indexed values.
  Example for feature `112`:

  ```python
  indexer['f']['112'] = [
    (1220239624, 10696),
    (1220122031, 9084),
    ...
  ]
  ```

---

### `item_feat_dict.json`

* Contains feature dictionaries for all items in the training set (useful for negative sampling).
* **Key:** item re-id
* **Value:** item feature dictionary (same format as `item_feature` in `seq.jsonl`).

---

### `seq_offsets.pkl`

* A Python list of byte offsets.
* `seq_offsets[i]` gives the file offset of line `i` in `seq.jsonl`.
* Used for efficient random access to user sequences and improved I/O performance.

**The examples are for illustrating the data format only and do not contain real user data.**