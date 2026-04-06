from __future__ import annotations
import json
import re
import random
import os

VOICE_TEMPLATES = [
    "A {gender} speaker delivers a {style} explanation in a clear teaching voice.",
    "This recording features a {gender} voice with a {style} speaking style.",
    "A {style} narration presented by a {gender} teacher.",
    "A {gender} educator explains the topic using a {style} tone.",
    "A clear and {style} explanation spoken by a {gender} voice.",
    "This audio contains a {gender} speaker using a {style} delivery.",
    "A {style} teaching narration performed by a {gender} individual.",
    "A professional {gender} voice speaking in a {style} manner.",
    "A calm and informative {style} explanation from a {gender} speaker.",
    "A {gender} teacher presents the topic with a {style} approach.",
    "A {style} educational explanation delivered by a {gender} voice.",
    "A natural {gender} voice expressing a {style} teaching style.",
    "A {gender} narrator speaks with a {style} tone for learning purposes.",
    "A friendly {style} explanation provided by a {gender} speaker.",
    "A focused {style} teaching voice from a {gender} educator.",
    "This sample includes a {gender} voice using a {style} narration style.",
    "A structured {style} explanation spoken by a {gender} teacher.",
    "A {gender} speaker communicates the lesson in a {style} way.",
    "An educational {style} narration performed by a {gender} voice.",
    "A confident {gender} speaker delivering a {style} explanation.",
    "A simple and {style} teaching narration from a {gender} educator.",
    "A {gender} instructional voice with a {style} expression.",
    "A clear {style} lesson explained by a {gender} speaker.",
    "A composed {gender} voice presenting content in a {style} tone.",
    "A {style} learning-focused narration by a {gender} teacher.",
    "A professional educational explanation in a {style} voice by a {gender} speaker.",
    "A {gender} speaker delivers knowledge using a {style} teaching tone.",
    "A smooth and {style} explanation spoken by a {gender} voice.",
    "A {style} classroom-style narration from a {gender} educator.",
    "A direct and {style} explanation presented by a {gender} speaker.",
    "An engaging {style} presentation given by a {gender} instructor.",
    "A {gender} narrator provides a {style} breakdown of the subject matter.",
    "The {gender} voice offers a {style} and academic delivery.",
    "A precise {style} lecture spoken by a {gender} academic.",
    "In a {style} manner, the {gender} speaker guides the listener through the topic.",
    "A highly articulate {gender} voice performing a {style} narration.",
    "The {style} quality of this {gender} speaker is perfect for educational content.",
    "A {gender} speaker uses an authoritative yet {style} tone.",
    "This {style} tutorial is narrated by a steady {gender} voice.",
    "A warm {gender} speaker provides a {style} instructional overview.",
    "The audio showcases a {gender} voice with a distinct {style} cadence.",
    "An articulate {style} explanation by a {gender} voice actor.",
    "A {gender} speaker adopts a {style} persona for this educational clip.",
    "This {style} delivery is performed by a clear-spoken {gender} individual.",
    "A {style} and methodical explanation from a {gender} speaker.",
    "The {gender} educator uses a {style} rhythm throughout the recording.",
    "A well-paced {style} narration delivered by a {gender} voice.",
    "A {gender} voice guides the lesson with a {style} and clear approach.",
    "The recording captures a {gender} speaker in a {style} teaching moment.",
    "A {style} and expressive {gender} voice recounts the educational material.",
    "This {gender} speaker provides a consistent {style} flow for learning.",
    "A balanced {style} tone is used by the {gender} narrator here.",
    "An insightful {style} explanation spoken by a {gender} specialist.",
    "The {gender} speaker maintains a {style} presence throughout the audio.",
    "A clear-cut {style} teaching style from a {gender} professional.",
    "This {gender} voice sounds both helpful and {style} in its delivery.",
    "A {style} pedagogical narration by a {gender} speaker.",
    "The {gender} speaker conveys complex ideas in a {style} tone.",
    "A rhythmic and {style} explanation given by a {gender} voice.",
    "This {style} auditory lesson is presented by a {gender} teacher.",
]


def _apply_voice_template(voice_description: str) -> str:
    """Convert 'female, cheerful' to a random template string."""
    try:
        parts  = [p.strip() for p in voice_description.split(",")]
        gender = parts[0]
        style  = parts[1]
        return random.choice(VOICE_TEMPLATES).format(gender=gender, style=style)
    except Exception:
        return voice_description


class StoryGenerator:

    def load_model(
        self,
        hf_token: str,
        model_id: str = "Qwen/Qwen2.5-32B-Instruct:featherless-ai",
        hf_base_url: str = "https://router.huggingface.co/v1",
        **kwargs,  # accepts and ignores legacy cache_dir / model_id kwargs
    ):
        """Initialise the HuggingFace Inference API client (OpenAI-compatible)."""
        from openai import OpenAI

        if not hf_token:
            raise RuntimeError("HF_TOKEN not set.")

        self.model_id = model_id
        self.client   = OpenAI(
            base_url=hf_base_url,
            api_key=hf_token,
        )
        print(f"HuggingFace API client ready — model: {self.model_id}")


    def _ask(self, prompt: str, max_new_tokens: int = 4000, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _extract_json(self, raw: str):
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = raw.find(start_char)
            if start == -1:
                continue
            end = raw.rfind(end_char)
            if end == -1:
                continue
            candidate = raw[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        for pattern in [r'\{.*\}', r'\[.*\]']:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        return None

    def _ask_json(self, prompt: str, max_new_tokens: int = 4000, temperature: float = 0.6):
        full_prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY valid JSON. "
            + "No extra text, no markdown, no code fences. Start directly with { or [."
        )
        last_raw = ""
        for attempt in range(5):
            raw = self._ask(full_prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            last_raw = raw
            print(f"JSON attempt {attempt + 1}, raw (first 200 chars): {raw[:200]}", flush=True)
            result = self._extract_json(raw)
            if result is not None:
                return result
            print(f"JSON parse attempt {attempt + 1} failed. Retrying...", flush=True)
        raise RuntimeError(
            f"Model did not return valid JSON after 5 attempts.\nLast output: {last_raw[:500]}"
        )


    def _normalize_gender(self, value: str) -> str:
        v = str(value).strip().lower()
        if v in ("male", "m", "boy", "man", "he", "him"):
            return "male"
        return "female"

    def _filter_voice_lines(self, lines: list, allowed_speakers: list) -> list:
        allowed_set = {s.lower() for s in allowed_speakers}
        filtered = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            speaker = line.get("speaker", "")
            if speaker.lower() not in allowed_set:
                print(f"Skipping unknown speaker '{speaker}'", flush=True)
                continue
            filtered.append(line)
        return filtered


    def analyze_lesson(self, lesson: str) -> dict:
        prompt = f"""You are an educational content designer for a children's animated show.
Your job is to break down a lesson into a story structure that teaches it clearly and accurately.

LESSON:
{lesson}

First, identify:
- How many distinct concepts or steps are in this lesson?
- Does the lesson involve interaction between things (e.g. sun + water + plant)?
- How many different locations or settings would naturally appear in this lesson?

Based on this, decide:
- num_characters: one character PER key concept or actor in the lesson (min 2, max 6)
- num_backgrounds: one background PER distinct setting in the lesson (min 2, max 6)
- num_scenes: one scene PER learning step (min 2, max 6)

CRITICAL: Every scene must teach a real, specific part of the lesson.

Return ONLY this JSON:
{{"num_characters": <int>, "num_backgrounds": <int>, "num_scenes": <int>,
  "reasoning": "<1 sentence why>",
  "lesson_steps": ["<step 1>", "<step 2>", "..."]}}"""
        return self._ask_json(prompt, max_new_tokens=600, temperature=0.3)

    def generate_characters(self, lesson: str, num_characters: int, lesson_steps: list) -> list:
        steps_text = "\n".join([f"- {s}" for s in lesson_steps])
        prompt = f"""You are creating characters for a children's educational animated story.

LESSON:
{lesson}

KEY LESSON STEPS:
{steps_text}

Generate EXACTLY {num_characters} characters.
Each character REPRESENTS a key concept or element from the lesson.

CRITICAL RULES:
- Characters MUST be objects or nature things from the lesson world (e.g. a sun, a raindrop, a cloud, a plant, a wind gust).
- Characters must NEVER be human or have a human face, human body, or human skin. NO people. NO humans. NO boys. NO girls.
- Each character's role must directly relate to what they represent in the lesson.
- Each character MUST have a different image_location so they do not overlap.

For each CHARACTER provide:
- name: ONE single word, fun and kid-friendly. ONE WORD ONLY.
- what_they_represent: the exact lesson concept this character embodies (1 sentence).
- gender: "male" or "female" only, lowercase.
- role: their specific educational role in the story (1 sentence).
- personality: what they are like (1 sentence).
- visual_description: A TTI image generation prompt for this character ONLY.
  Rules:
  * The character MUST be a cartoon version of a NON-HUMAN object or nature element (e.g. a cartoon sun with a smiley face, a cartoon water droplet, a cartoon cloud). NEVER a human, person, boy, or girl.
  * Pure white background - character only, no scene, no environment.
  * Strictly 2D cartoon style, bold black outlines, flat bright colors. NO photorealism. NO 3D rendering. NO human anatomy.
  * Give the character a cute cartoon face: large round expressive eyes, a wide animated smiling mouth. NO human nose. NO human ears. NO human skin.
  * End with: "pure white background, 2D cartoon illustration, children's TV show style, no humans, no people, no realistic features".
  * 3-4 sentences total.
    - image_location: Where this character will be placed on the scene image.
      Return as a JSON object: {{"x": <float>, "y": <float>}}
      where (0.0, 0.0) = top-left, (1.0, 1.0) = bottom-right.
      CRITICAL: Use x values between 0.1 and 0.9 and y values between 0.2 and 0.8 to keep characters safely within the frame.
      Each character MUST have a DIFFERENT position.
- voice_description: ONLY two things separated by a comma:
  (1) gender using ONLY "male" or "female" in lowercase,
  (2) ONE emotion word from ONLY this list:
      cheerful, gentle, energetic, whispering, authoritative,
      playful, calm, excited, curious, friendly, enthusiastic, soothing, animated, bright
  Example: "female, cheerful"

Return a JSON array of EXACTLY {num_characters} objects."""
        chars = self._ask_json(prompt, max_new_tokens=1500 * num_characters)
        if isinstance(chars, list):
            for c in chars:
                if isinstance(c, dict):
                    c["gender"] = self._normalize_gender(c.get("gender", "female"))
        return chars

    def generate_backgrounds(self, lesson: str, num_backgrounds: int, lesson_steps: list) -> list:
        steps_text = "\n".join([f"- {s}" for s in lesson_steps])
        prompt = f"""You are creating background scenes for a children's educational animated story.

LESSON:
{lesson}

KEY LESSON STEPS:
{steps_text}

Generate EXACTLY {num_backgrounds} background scenes.
Each background must reflect a REAL setting where part of this lesson takes place.
Do NOT include any characters in the background.

For each BACKGROUND provide:
- name: short 2-3 word label
- lesson_context: which part of the lesson happens here (1 sentence)
- visual_description: VERY DETAILED background-only description. NO characters. NO humans. NO people.
  Style rules: strictly 2D cartoon illustration style, flat bright colors, bold outlines, children's animated TV show look.
  Must end with: "2D cartoon background, children's animated show style, no characters, no people, no humans".
  3-4 sentences total.
- mood: one word (cheerful, cozy, adventurous, mysterious, warm, bright, etc.)

Return a JSON array of EXACTLY {num_backgrounds} background objects:
[{{"name":"","lesson_context":"","visual_description":"","mood":""}}]"""
        return self._ask_json(prompt, max_new_tokens=1200 * num_backgrounds)

    def generate_outline(self, lesson: str, characters: list, backgrounds: list,
                         num_scenes: int, lesson_steps: list) -> list:
        chars_summary = "\n".join([
            f"- {c['name']} ({c['gender']}): represents {c.get('what_they_represent', c['role'])}"
            for c in characters
        ])
        bgs_summary = "\n".join([
            f"- {b['name']}: {b.get('lesson_context', b['mood'])}"
            for b in backgrounds
        ])
        steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(lesson_steps)])
        bg_names   = ", ".join([b['name'] for b in backgrounds])
        char_names = ", ".join([c['name'] for c in characters])

        prompt = f"""You are writing a children's educational story outline.
The story exists ONLY to teach this lesson - every scene must serve the lesson.

LESSON:
{lesson}

LESSON STEPS TO COVER (every step must appear in at least one scene):
{steps_text}

CHARACTERS:
{chars_summary}

AVAILABLE BACKGROUNDS:
{bgs_summary}

Create EXACTLY {num_scenes} scenes. Rules:
- Map each lesson step to one or more scenes - cover ALL steps, in order
- Every scene must have a clear "lesson_element" that is a specific accurate fact from the lesson
- Only use character names from: {char_names}
- Only use background names from: {bg_names}

Return a JSON array of exactly {num_scenes} scene objects:
[
  {{
    "scene_number": 1,
    "title": "",
    "background": "",
    "characters_present": [
      {{"name": "<character name>", "position": {{"x": 0.0, "y": 0.0}}}}
    ],
    "event": "",
    "lesson_element": "<specific accurate fact from the lesson>",
    "lesson_step_covered": "<which lesson step this scene covers>"
  }}
]"""
        return self._ask_json(prompt, max_new_tokens=4000)

    def generate_passages(self, lesson: str, outline: list,
                          characters: list, backgrounds: list,
                          lesson_steps: list) -> list:

        def get_char_names(characters_present):
            names = []
            for entry in characters_present:
                if isinstance(entry, dict):
                    names.append(entry['name'])
                elif isinstance(entry, str):
                    names.append(entry)
            return names

        def get_bg(name):
            for b in backgrounds:
                if b['name'].lower().strip() == name.lower().strip():
                    return b['visual_description']
            return name

        def get_profiles(characters_present):
            names = get_char_names(characters_present)
            lines = []
            for n in names:
                for c in characters:
                    if c['name'] == n:
                        lines.append(
                            f"{c['name']} ({c['gender']}): "
                            f"represents {c.get('what_they_represent', c['role'])}. "
                            f"Personality: {c['personality']}."
                        )
            return "\n".join(lines)

        steps_text   = "\n".join([f"- {s}" for s in lesson_steps])
        passages     = []
        story_so_far = ""

        for scene in outline:
            bg_detail       = get_bg(scene['background'])
            char_profiles   = get_profiles(scene['characters_present'])
            char_names_list = get_char_names(scene['characters_present'])
            num_chars       = len(char_names_list)
            interact_rule   = (
                "- Characters MUST talk to each other and their dialogue must explain "
                "how their concepts relate to each other in the lesson"
                if num_chars > 1 else ""
            )
            prompt = f"""Write one passage of a children's educational story.

FULL LESSON BEING TAUGHT:
{lesson}

ALL LESSON STEPS:
{steps_text}

THIS SCENE TEACHES: {scene['lesson_element']}
LESSON STEP COVERED: {scene.get('lesson_step_covered', '')}

Story so far:
{story_so_far if story_so_far else "(This is the first scene.)"}

Scene event: {scene['event']}
Background: {bg_detail}

Characters in this scene: {", ".join(char_names_list)}
{char_profiles}

CRITICAL WRITING RULES:
- The passage MUST clearly teach: "{scene['lesson_element']}"
- Characters speak and act AS their lesson concept
- Use simple words and short sentences (age 5-8)
- Cheerful and warm tone
- Length: 4-6 short paragraphs
- ONLY use characters listed above
- Return ONLY the story text, no titles or labels
{interact_rule}"""
            text = self._ask(prompt, max_new_tokens=2000, temperature=0.6)
            passages.append({
                "scene_number":           scene['scene_number'],
                "title":                  scene['title'],
                "background":             scene['background'],
                "background_description": bg_detail,
                "characters_present":     scene['characters_present'],
                "lesson_element":         scene['lesson_element'],
                "lesson_step_covered":    scene.get('lesson_step_covered', ''),
                "passage":                text,
            })
            story_so_far += f"\n\n[Scene {scene['scene_number']} - taught: {scene['lesson_element']}]\n{text}"
            print(f"Scene {scene['scene_number']} written.", flush=True)
        return passages

    def generate_voice_scripts(self, passages: list, characters: list) -> list:
        all_scripts = []
        char_lookup = {c['name']: c for c in characters}

        for sp in passages:
            scene_char_entries = sp['characters_present']
            scene_pos          = {}
            char_names_list    = []

            for entry in scene_char_entries:
                if isinstance(entry, dict):
                    char_names_list.append(entry['name'])
                    scene_pos[entry['name']] = entry.get('position', {})
                elif isinstance(entry, str):
                    char_names_list.append(entry)
                    scene_pos[entry] = {}

            char_voice_info = "\n".join([
                f"- {name} (gender: {char_lookup[name]['gender']}): {char_lookup[name]['voice_description']}"
                for name in char_names_list if name in char_lookup
            ])

            # Build a lookup of what each character represents (their real-world identity)
            char_identity_map = "\n".join([
                f"- {name} = {char_lookup[name].get('what_they_represent', char_lookup[name].get('role', name))}"
                for name in char_names_list if name in char_lookup
            ])

            location_map = "\n".join([
                f"- {name}: x={scene_pos[name].get('x','?')}, y={scene_pos[name].get('y','?')}"
                for name in char_names_list
            ])

            # Build identity descriptions for use in video_prompt context
            identity_lines = [
                f"{name} (who represents {char_lookup[name].get('what_they_represent', char_lookup[name].get('role', name))})"
                for name in char_names_list if name in char_lookup
            ]

            others_context = (
                f"Characters in this scene: {', '.join(identity_lines)}. "
                "For each shot, one character speaks while the others are still present and visible. "
                "The video_prompt MUST clearly identify the speaking character by BOTH their name AND what they represent "
                "(e.g. 'Sunshine the sun', 'Droplet the water droplet', 'Breezy the wind'). "
                "It must also state their role (e.g. 'who provides energy to plants'). "
                "CRITICAL: To prevent the AI from animating the wrong character, you MUST explicitly state that all other characters are 'completely static and unmoving'. "
                "Describe what EVERY other character in the scene is doing (e.g. 'watching attentively', 'listening', 'looking curious') but always include the phrase 'completely static and frozen' for them. "
                "Identify each of them by name and what they represent."
                if len(char_names_list) > 1
                else
                f"The only character in this scene is {char_names_list[0]}, who represents "
                f"{char_lookup[char_names_list[0]].get('what_they_represent', char_lookup[char_names_list[0]].get('role', char_names_list[0])) if char_names_list[0] in char_lookup else char_names_list[0]}. "
                "The video_prompt should identify this character by both their name and what they represent, "
                "mention their role, and focus entirely on this character speaking."
            )

            # Build position hints from x,y coords
            def pos_hint(name):
                pos = scene_pos.get(name, {})
                x = float(pos.get("x", 0.5))
                y = float(pos.get("y", 0.5))
                h = "left" if x < 0.4 else ("right" if x > 0.6 else "center")
                v = "top" if y < 0.4 else ("bottom" if y > 0.6 else "middle")
                return f"{v} {h}"

            pos_hints = "\n".join([
                f"- {name}: {pos_hint(name)} of the frame"
                for name in char_names_list
            ])

            prompt = f"""You are writing a voice script for a children's educational animated video.

THIS SCENE TEACHES: {sp['lesson_element']}
BACKGROUND: {sp['background_description']}

CHARACTER POSITIONS IN FRAME:
{pos_hints}

CHARACTER VOICES:
{char_voice_info}

{others_context}

CRITICAL RULES:
- Every line of dialogue MUST relate to the lesson fact being taught
- NO narration - only character dialogue
- Each line must move the lesson understanding forward
- Keep dialogue natural and fun

LINE LENGTH RULES:
- Every "text" field MUST be between 38 and 50 words. Count carefully.
- NEVER write fewer than 38 words or more than 50 words per line.
- This ensures each line takes between 15 and 20 seconds to speak aloud.

Story passage to adapt into dialogue:
{sp["passage"]}

Return a JSON array. Each item must have EXACTLY these 4 fields:
1. speaker: MUST be exactly one of: {", ".join(char_names_list)}
2. text: EXACTLY 38 to 50 words of natural dialogue (this must take 15-20 seconds to speak).
3. voice_description: ONLY gender and ONE emotion word separated by a comma.
   Example: "female, cheerful"
4. video_prompt: MUST follow this EXACT structure sentence by sentence:
   Sentence 1: "A beautiful landscape with [what char1 represents] and [what char2 represents]."
   Sentence 2: "[What the STATIC character represents, capitalized] in the [their position] remains completely static, its eyes wide and its smile frozen."
   Sentence 3: "[What the SPEAKING character represents, capitalized] in the [their position] is the only character moving."
   Sentence 4: "[What the SPEAKING character represents, capitalized] speaks the words in the voice, its mouth opening and closing in perfect synchronization."
   Sentence 5: "[What the SPEAKING character represents, capitalized]\'s eyes blink and its body [a natural movement word matching their type, e.g. ripples/shimmers/puffs/glows] as it talks."

   STRICT RULES:
   - Use ONLY what each character represents (e.g. "the sun", "the water droplet", "the cloud") — NEVER use character names.
   - Use CHARACTER IDENTITY MAP for what each character represents.
   - Use POSITION MAP for frame positions.
   - Follow the 5 sentences in EXACT order — do not add or remove sentences.
   - If there is only one character in the scene, skip sentence 2.

5. negative_prompt: MUST follow this EXACT structure:
   "[what the STATIC character represents] speaking, [what the STATIC character represents] mouth moving, [what the STATIC character represents] animating, [what the STATIC character represents] blinking, [what the STATIC character represents] changing expression"

   STRICT RULES:
   - Use ONLY what the static character represents (e.g. "the sun", "the water droplet") — NEVER use character names.
   - If there is only one character in the scene, set negative_prompt to "".

CHARACTER IDENTITY MAP (name → what they represent):
{char_identity_map}

CHARACTER POSITION MAP:
{pos_hints}

Return ONLY the JSON array."""

            lines = self._ask_json(prompt, max_new_tokens=5000, temperature=0.4)
            lines = self._filter_voice_lines(lines, char_names_list)

            for line in lines:
                if "voice_description" in line and line["voice_description"]:
                    line["voice_description"] = _apply_voice_template(
                        line["voice_description"]
                    )

            all_scripts.append({
                "scene_number":       sp['scene_number'],
                "title":              sp['title'],
                "background":         sp['background'],
                "characters_present": scene_char_entries,
                "lesson_element":     sp['lesson_element'],
                "script":             lines,
            })
            print(f"Scene {sp['scene_number']} scripted ({len(lines)} lines).", flush=True)
        return all_scripts