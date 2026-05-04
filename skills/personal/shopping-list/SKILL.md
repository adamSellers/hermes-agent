---
name: shopping-list
description: Track shopping lists across sessions. Add, remove, mark items as bought, and view the current list when heading to the shops.
version: 1.1.0
author: Adam
license: MIT
metadata:
  hermes:
    tags: [Productivity, Personal, Lists, Shopping]
    related_skills: []
    requires_tools: [recall_memory, remember, forget]
---

# Shopping List

Persistent shopping lists backed by Hermes durable memory. This skill owns
small state records, so it must write them explicitly with `remember`. Do not
rely on the background memory extractor to infer shopping-list state from
assistant prose.

## When to Use

Load this skill when the user:

- Mentions running out of something or needing to buy something
- Asks to see their shopping list, grocery list, or what they need to buy
- Says they are heading to the shops, supermarket, Woolies, Coles, Bunnings,
  pharmacy, etc.
- Reports buying or picking up an item that was on the list
- Asks to clear, reset, or remove items from the list

Do not load for meal planning, recipe lookup, or budgeting.

## Storage Model

Each shopping-list item is one durable `fact` memory.

Use this shape:

```json
{
  "kind": "fact",
  "content": {
    "subject": "shopping_list:<list>:<item_slug>",
    "subject_type": "thing",
    "predicate": "shopping_item_status",
    "object": "status=<status> item=<item> list=<list> category=<category>"
  },
  "tags": [
    "shopping_list",
    "shopping_list:<list>",
    "status:<status>",
    "category:<category>"
  ],
  "confidence": 0.95
}
```

Rules:

- Default `list` is `groceries`.
- `status` is one of `pending`, `bought`, `removed`.
- Omit `category=<category>` and `category:<category>` if unclear.
- `item_slug` is lowercase ASCII made from the item string with spaces and
  punctuation changed to underscores. Example: `dried chick peas` becomes
  `dried_chick_peas`.
- For the same item, always reuse the same `subject` and predicate. Writing a
  new status supersedes the old active fact, so pending items do not remain
  visible after they are bought or removed.

## Add Item

When the user wants to add an item, call `remember` immediately.

Example for "add dried chick peas to the list":

```json
remember({
  "kind": "fact",
  "content": {
    "subject": "shopping_list:groceries:dried_chick_peas",
    "subject_type": "thing",
    "predicate": "shopping_item_status",
    "object": "status=pending item=dried chick peas list=groceries"
  },
  "tags": ["shopping_list", "shopping_list:groceries", "status:pending"],
  "confidence": 0.95
})
```

Then reply briefly: "Added dried chick peas to your groceries list."

## Show Current List

When the user asks what is on the list, call:

```json
recall_memory({
  "query": "pending",
  "tags": ["shopping_list:groceries", "status:pending"],
  "kinds": ["fact"],
  "max_results": 100
})
```

Then:

1. Parse `item=...`, `category=...`, and `list=...` from each summary.
2. Ignore anything without `status=pending`.
3. Group by category if present; otherwise show one flat list.
4. Sort categories and items alphabetically.
5. Render as a checklist using `[ ]`.
6. If recall returns no durable memory results, say "Your groceries list is empty."

## Mark Bought

First recall pending candidates:

```json
recall_memory({
  "query": "<item>",
  "tags": ["shopping_list:<list>", "status:pending"],
  "kinds": ["fact"],
  "max_results": 20
})
```

If there is one clear match, call `remember` again with the same `subject` and
predicate, but with:

- `object`: `status=bought item=<canonical item> list=<list> ...`
- `tags`: `["shopping_list", "shopping_list:<list>", "status:bought", ...]`

If there are multiple plausible matches, ask which one.

## Remove Without Buying

Use the same flow as Mark Bought, but write `status=removed` and tag
`status:removed`.

## Clear The List

If the user says "clear the list" after shopping, recall bought items with
`tags=["shopping_list:<list>", "status:bought"]` and mark each as `removed`.
Leave still-pending items alone.

If the user says "clear everything", confirm first. Then mark pending and
bought items as `removed`.

## List Shopping Lists

Call:

```json
recall_memory({
  "query": "shopping list",
  "tags": ["shopping_list"],
  "kinds": ["fact"],
  "max_results": 100
})
```

Pull distinct list names from tags matching `shopping_list:<list>` and return
them sorted with pending counts.

## Pitfalls

- Do not write `Shopping list add: ...` and assume extraction will save it.
  Assistant prose is not a reliable storage API.
- Do not invent `memory_set`; the write tool is `remember`.
- Do not create multiple active pending records for the same item. Reuse the
  same `subject` and `shopping_item_status` predicate so status changes
  supersede the old active fact.
- Do not claim the list is empty if recall returns an error. Say recall failed.

## Verification

After each distinct user request, verify the changed state with `recall_memory`
using the relevant list and status tags.
