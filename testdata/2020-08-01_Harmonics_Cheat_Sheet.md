# Harmonics Cheat Sheet

## Classic ABCD Pattern

**Definition: Classic ABCD Pattern** - The simplest harmonic pattern. It is very basic and best to be traded when it is complete (at the point D). SL should be based on MS not Fibs.

> **Pro Tip:** If C+.618 look for 1.68; if C+.786 look for 1.272. fib time= 1

> **Warning:** Invalidation: 1.618. Targets: .382 and .618

### 📊 Chart Analysis: Classic ABCD Pattern Example

**Metadata:** Asset: NEO/USD, Timeframe: [UNCLEAR], Indicator Parameters: Fibonacci retracement and extension levels

**Spatial Annotation Map:**
- Point A at price level 11.17
- Point B at price level 8.43 (0.0 Fibonacci level)
- Point C at 0.618 retracement (10.12) and 0.786 retracement (10.59)
- Point D at the target level (1.64)
- "Target: 1.64(19.25%) 164, Amount: 2138.16" label positioned at the top of the chart
- "Stop: 0.36(4.23%) 36, Amount: 750.16" label positioned below the target

**The Narrative Flow (Chronological):**
Step 1: Price starts at point A (11.17). Step 2: Price moves down to point B (8.43). Step 3: Price retraces to point C (10.12 at 0.618 level). Step 4: Price moves down to point D (1.64 extension level).

**Hard Data Extraction:**
- A: 11.17
- B: 8.43
- 0.886: 10.86
- 0.786: 10.59
- 0.65: 10.21
- 0.618: 10.12
- 0.5: 9.80
- 0.382: 9.48
- 0.236: 9.07
- Target: 1.64 (19.25%)
- Stop: 0.36 (4.23%)
- Amount: 2138.16 (Target)
- Amount: 750.16 (Stop)

```yaml
Strategy Name: Classic ABCD Pattern
Direction: Short (in example)
Conditions:
  - IF: Price completes ABCD pattern with C at 0.618-0.786 retracement of AB
  - AND: If C+.618 look for 1.618 extension; if C+.786 look for 1.272 extension
Triggers:
  - ENTRY: At point D (completion of pattern)
  - STOP LOSS: Beyond 1.618 extension of XA
  - TAKE PROFIT: .382 and .618 retracement levels
```

## Butterfly Harmonic

**Definition: Butterfly Harmonic** - An external harmonic pattern where the retracement of XA has to be heavy, it has to reach .786.

> **Pro Tip:** Volume plays a big role here. XA: You want to see the impulse going up on declining volume. AB: Increasing Bear volume. BC: Declining volume. CD: If volume keeps increasing, don't trade the D and stay in trade.

> **Warning:** Invalidation: 1.618 of XA. Targets: X, B, CC of whole move measured from A

```yaml
Strategy Name: Butterfly Harmonic
Direction: [Not specified in document - depends on pattern orientation]
Conditions:
  - IF: B retraces to .786 of XA
  - AND: C is between .382 and .886 of AB (with .618 preferred)
  - AND: BC expansion = 1.618-2.618 of AB
  - AND: XA expansion = 1.27-1.618
  - AND: AB extension = 1
Triggers:
  - ENTRY: At point D (completion of pattern)
  - STOP LOSS: Beyond 1.618 of XA
  - TAKE PROFIT: X, B, and Confluence Point (CC) of whole move measured from A
```

## Crab Harmonic

**Definition: Crab Harmonic** - Very similar to Butterfly pattern but the D leg goes deeper. It is also an external pattern.

> **Pro Tip:** The preferred retracement of XA is .618 and between .5 and .618 for the AB retracement.

> **Warning:** Invalidation: 1.618 of XA. Targets: CC of C, Top of A

```yaml
Strategy Name: Crab Harmonic
Direction: [Not specified in document - depends on pattern orientation]
Conditions:
  - IF: B retraces to .382-.618 of XA (.618 preferred)
  - AND: C is between .382-.886 of AB (between .5 and .618 preferred)
  - AND: BC expansion = 1.618-2.618 of AB
  - AND: D = 2.24-3.618 of XA (3.14 preferred)
  - AND: AB extension = 1.618
Triggers:
  - ENTRY: At point D (completion of pattern)
  - STOP LOSS: Beyond 1.618 of XA
  - TAKE PROFIT: Confluence Point (CC) of C, Top of A
```

## Bat Harmonic

**Definition: Bat Harmonic** - An internal consolidation pattern where it is mandatory that point B retraces strictly between .382 and .5 of XA.

> **Pro Tip:** A 'perfect' BAT retraces to .5 (XA and AB) and is even more likely to play out. It also depends on the market you are trading – stock markets are more likely to retrace to .5 while crypto markets tend to retrace to .618 more.

> **Warning:** Invalidation: X. Targets: .618 and A

```yaml
Strategy Name: Bat Harmonic
Direction: [Not specified in document - depends on pattern orientation]
Conditions:
  - IF: B retraces to .382-.5 of XA (with .5 being "perfect")
  - AND: C is at .886 retracement of AB
  - AND: D is between 1.618-2.618 of BC expansion
  - AND: AB extension = 1
Triggers:
  - ENTRY: At point D (completion of pattern)
  - STOP LOSS: Beyond X (origin point)
  - TAKE PROFIT: .618 retracement level and Point A
```

## Gartley Harmonic

**Definition: Gartley Harmonic** - An internal consolidation harmonic pattern which is similar to BAT pattern. The big difference is that Gartley retraces to a 'perfect' .618 (XA and AB).

> **Pro Tip:** They are very common, you literally see them every day.

> **Warning:** Invalidation: X. Targets: .618 OR 1-1 from X to A to D

```yaml
Strategy Name: Gartley Harmonic
Direction: [Not specified in document - depends on pattern orientation]
Conditions:
  - IF: B retraces to .618 of XA
  - AND: C retraces to .618 of AB
  - AND: D is at .786 retracement of XA
  - AND: D is between 1.27-1.613 of BC expansion
Triggers:
  - ENTRY: At point D (completion of pattern)
  - STOP LOSS: Beyond X (origin point)
  - TAKE PROFIT: .618 retracement level OR 1-1 extension from X to A to D
```

## Cypher Harmonic

**Definition: Cypher Harmonic** - The only harmonic that uses Fib extension. It is unique and simple to use. What is unique is that to get C – taking the fib extension you click on X, A and X again.

> **Warning:** Invalidation: x. Targets: Point A (Maybe B) and CC from CD

```yaml
Strategy Name: Cypher Harmonic
Direction: [Not specified in document - depends on pattern orientation]
Conditions:
  - IF: B retraces to .382-.618 of XA
  - AND: C is between 1.133-1.414 extension of XA
  - AND: D is at .786 retracement of XC (B cannot hit this level)
  - AND: D is at 1.272-2.0 extension of BC
Triggers:
  - ENTRY: At point D (completion of pattern)
  - STOP LOSS: Beyond X (origin point)
  - TAKE PROFIT: Point A (and possibly Point B), Confluence Point (CC) from CD
```

## Shark Harmonic

**Definition: Shark Harmonic** - A very nice harmonic that can be seen on BTC a lot. One can incorporate C with the SFP. It is unique in a way that B can retrace anywhere between .236 and .886.

> **Pro Tip:** Wave D is usually very impulsive. The shark harmonic implies a strong impulsive move up, especially if the C comes off a swing failure of the A. You can trade the C up to the D, with D being a big target. If the D is invalidated, you are likely in a wave 3.

> **Warning:** Invalidation: 1.133 of X. Targets: 0.5 biggest target. Keep 5-0 in mind. Tp2= C

```yaml
Strategy Name: Shark Harmonic
Direction: [Not specified in document - depends on pattern orientation]
Conditions:
  - IF: B retraces anywhere between .236-.886 of XA (.382 is common)
  - AND: C is between 1.13-1.618 of AB expansion
  - AND: BC expansion = 1.618-2.24
  - AND: XA expansion = .886-1.13
Triggers:
  - ENTRY: At point C (for the move to D) or at point D
  - STOP LOSS: Beyond 1.133 of X
  - TAKE PROFIT: .5 retracement level (main target), Point C (secondary target)
```

## 5-0 (Next Step from the Shark) Harmonic

**Definition: 5-0 Harmonic** - It continues from the Shark pattern, starting with shorting the D of it. The target is, as the name suggests, .5 fib and it should bounce right there to be valid.

> **Pro Tip:** When a shark bounces and reverses off the 0.5 fib taken from CD, it turns into a 5-0 pattern.

> **Warning:** Invalidation: .886 of CD

```yaml
Strategy Name: 5-0 Harmonic
Direction: [Not specified in document - depends on pattern orientation]
Conditions:
  - IF: Shark pattern completes at D
  - AND: Price reverses from D and moves toward .5 retracement of CD
  - AND: Price bounces from .5 retracement level
Triggers:
  - ENTRY: At point E (0.5 retracement of CD)
  - STOP LOSS: Beyond .886 of CD
  - TAKE PROFIT: Reversal point at E (0.5 retracement)
```