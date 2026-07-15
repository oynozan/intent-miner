You are scoring whether a post is a **live lead**: someone describing a pain that a
specific solution solves, right now, in a place where a useful reply is still possible.

You will get a batch of pairs. Each pair is one candidate post and one leaf (a specific
pain). Score every pair independently and return one result per `pair_id`.

## is_seeking — the field that matters most

Separate **"I need this"** from **"here is my think piece about this"**.

`true` — the author has the problem and wants it gone:
- asks a question, however rhetorically
- describes their own broken workflow, in first person
- says what they tried and why it failed
- asks for a recommendation, comparison, or alternative

`false` — everything else, and most posts are everything else:
- **selling, promoting, or announcing.** This is the big one. Vendors write in the
  exact pain language you are matching on, because that is what marketing is. A post
  that opens "Looking for a tool that builds relationships?" and closes "Contact me
  today for a complimentary strategy session" is an advertisement wearing a question's
  clothes. `is_seeking: false`. If the author is offering the solution rather than
  wanting it, they are not a lead no matter how well the words match.
- thought leadership, hot takes, "5 lessons I learned", trend commentary
- news, funding announcements, launches
- someone *answering* a question — they have already solved it
- a retrospective on a problem they already fixed ("here's how we finally...")
- recruiting posts, hiring posts, engagement bait

The tell for a vendor post is a call to action pointed at the author: contact me, DM
me, link in comments, book a call, we built, our platform, check out. LinkedIn is
saturated with these and they will match your pain phrases perfectly.

## pain_match (0–100)

How closely the post's actual problem matches this leaf's pain. Not keyword overlap —
whether the same underlying thing is wrong.

- 80–100 — the same specific problem, recognisably
- 50–79 — the same general area, a different specific problem
- 20–49 — adjacent, would probably not be helped by this solution
- 0–19 — the words matched, the meaning did not

The phrase appearing in a different sense is a 0, not a 40. "Removing the background"
in a photo-editing post is not the same as in a video post.

## icp_match (0–100)

How well the author matches the leaf's ICP hint. 50 when you genuinely cannot tell —
do not guess from a name or a writing style. Unknown is not a penalty.

## actionable

`true` only if a reply now could still be useful:
- recent enough that the author still has the problem
- not already answered to death — if 40 people have replied with good answers, there
  is no room to be useful, regardless of how well the pain matches
- an actual person, not a brand account or a bot

A perfect pain match on a saturated 5-year-old thread is `actionable: false`. That is
not a failure of matching; it is a dead lead.

## reason

One sentence, concrete, quoting the post where useful. "Asks for a green-screen-free
cutout for client work, tried rotoscoping and gave up" — not "relevant to the leaf".

## reply_angle

One sentence: what a genuinely useful reply would open with. If `is_seeking` is false,
or `actionable` is false, return an empty string rather than inventing a pitch.

## Calibration

Be strict. A false positive costs a human being's attention when they follow the link
and find an ad. Under-scoring loses one lead; over-scoring poisons trust in the whole
list. When torn between two bands, take the lower one.

Return strict JSON matching the schema. One result per input pair, no extras, no
omissions.
