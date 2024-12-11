fs_examples = """
<example>
Type: Neutral_with_less_intensity
Polar theta:  42.616
Polar radius:  0.169
Review:  Over priced!  But then again it is the Las Vegas strip!! Cool atmosphere, typical food.
</example>
<example>
Type: Neutral_with_less_intensity
Polar theta:  46.44
Polar radius:  0.117
Review:  Quick food.  Not surprisingly it tastes like a chain
</example>
<example>
Type: Neutral_with_negligible intensity
Polar theta:  42.79
Polar radius:  0.1423
Review:  no comment!
</example>
<example>
Type:  neutral_with_more_intensity
Polar theta:  43.300
Polar radius:  1.059
Review:  Great food with great vegan/vegetarian options . The staff could work on customer service as well as efficiency.  I literally dock a full star because their front door was getting stuck on the cement outside creating this awful nails on a chalkboard sound. No one seemed to want solve the problem. When my boyfriend attempted to prop the door so it wouldn't make that sound, the worker shut it again creating a anxious and very interrupted dining experience.
</example>
<example>
Type:  neutral_with_more_intensity
Polar theta:  43.3817
Polar radius:  1.0117
Review:  So I went back to Casbah for dinner and was totally underwhelmed. The granola I had for brunch (in my previous review) was better than my entire dinner put together. The service was better at dinner then at brunch, but the food made my rating drop from five stars (probably a little generous) to three stars. Definitely go for brunch! But it's not worth the money for dinner. Mediocre food in a nice atmosphere with good service is still mediocre food...
</example>
<example>
Type:  neutral_with_more_intensity
Polar theta:  47.7955
Polar radius:  0.9205
Review:  I headed over to here after roller derby shenanigans one Thursday night. Thursdays at Hartigans are karaoke night so I was kept pretty entertained.\n\nThe interior is nice-- almost sport-bar-y by feel rather than a pub. Brightly lit, and more tables and bar seats than booths. \n\nThe bartender wasn't exactly rude to me, but not particularly nice either.\n\nI snacked on some fries while I was there- pretty subpar. The beer selection was mediocre and definitely a pinch more pricey than the quality I got.\n\nOne great thing though, I have to mention, about this place is it is GLBT friendly and not flamboyant about it. In the south, that isn't as common so yeah. Props for that.
</example>
<example>
Type:  neutral_with_more_intensity
Polar theta:  37.604
Polar radius:  0.889
Review:  All I can say is the worst! We were the only 2 people in the place for lunch, the place was freezing and loaded with kids toys! 2 bicycles, a scooter, and an electronic keyboard graced the dining room. A fish tank with filthy, slimy fingerprints smeared all over it is there for your enjoyment.\n\nOur food came... no water to drink, no tea, medium temperature food. Of course its cold, just like the room, I never took my jacket off! The plates are too small, you food spills over onto some semi-clean tables as you sit in your completely worn out booth seat. The fried noodles were out of a box and nasty, the shrimp was mushy, the fried rice was bright yellow.\n\nWe asked for water, they brought us 1 in a SOLO cup for 2 people. I asked for hot tea, they said 10 minutes. What Chinese restaurant does not have hot tea available upon request?\n\nOver all.... my first and last visit to this place. The only good point was that it was cheap, and deservingly so.
</example>
<example>
Type:  positive_with_less_intensity
Polar theta:  66.885
Polar radius:  0.312
Review:  Good desserts and nice cool, calm atmosphere!
</example>
<example>
Type:  positive_with_less_intensity
Polar theta:  61.7602
Polar radius:  0.30661
Review:  This spot is legit. Great clothing, friendly and knowledgable staff and convenient location.
</example>
<example>
Type:  positive_with_more_intensity
Polar theta:  71.252
Polar radius:  0.4609
Review:  Love their pizza!!! Always fresh & made to order!  Try the warm chocolate chip cookie with ice cream :)
</example>
<example>
Type:  positive_with_more_intensity
Polar theta:  70.931
Polar radius:  0.435
Review:  Their Kobe stake melts in your  mouth...absolutely phenomenal....loved loved loved their truffle fries! Great drinks, wonderful service. I will definitely stop here again next time I'm in town.
</example>
<example>
Type:  negative_with_less_intensity
Polar theta:  38.847
Polar radius:  0.1839
Review:  Meh.  Had the filet and it was a little bland.  My salmon was very fishy.
</example>
<example>
Type:  negative_with_less_intensity
Polar theta:  38.457756107063446
Polar radius:  0.2007546268571123
Review:  The food was good, but was way over-priced (reminded me of Dardanelles on Monroe Street). \n\nDecor is cool though.
</example>
<example>
Type:  negative_with_more_intensity
Polar theta:  38.24163679682066
Polar radius:  1.3860592929365827
Review:  Waitress. the turkey burger is cold. Can you recook? We brought another. Ah, look, it's pink and uncooked. Sorry, we are learning the grill. That's my review. I hope it's better for you. Food is not a priority.
</example>
<example>
Type:  negative_with_more_intensity
Polar theta:  28.783420645493035
Polar radius:  0.7598648254190514
Review:  Customer service was poor. And the smoothie I got was so awful I had a horrible taste in my mouth for the longest time.
</example>
<example>
Type:  negative_with_more_intensity
Polar theta:  28.677162677979627
Polar radius:  0.4956996921593206
Review:  They won't have anything hot and ready. Slow everything. Save your receipt, because they won't remember your order. Check your order before leaving. Expect to wait 30 minutes for anything.\n\nSeriously, I wish I could give them negative stars. Just the worst.
</example>
"""


polar_gen_prompt = f"""
You are a restaurant review generator that creates reviews based on polar coordinates representing sentiment (θ) and intensity (r). Given specific θ and r values, generate authentic restaurant reviews that match these parameters.

SENTIMENT MAPPING (θ):
- 0° = Extremely negative
- 45° = Neutral
- 90° = Extremely positive

INTENSITY MAPPING (r):
- 0.0-0.3: Minimal emotional investment, brief comments
- 0.3-0.7: Moderate detail and emotional expression
- 0.7-1.0: Strong opinions, detailed experiences, multiple aspects
- >1.0: Very intense reactions with extensive detail

KEY PATTERNS:
1. Low Intensity (r < 0.3)
   - Use brief, simple statements
   - Focus on one aspect
   - Minimal punctuation
   - Example: "Decent food" (θ=45°, r=0.1)

2. Medium Intensity (0.3 < r < 0.7)
   - 2-3 sentences
   - Balance of details
   - Moderate emotional language
   - Example: "Good food but service was slow" (θ=45°, r=0.5)

3. High Intensity (r > 0.7)
   - Multiple aspects covered
   - Strong emotional language
   - Specific examples and experiences
   - More punctuation (!!!)
   - Personal stories
   - Example: "Amazing food! The chef came out to greet us!" (θ=75°, r=0.9)

Below are some examples:
{fs_examples}

Generate a restaurant review matching these coordinates. Enclose the review in <review></review> tags. Remember to be authentic and creative in your responses!
"""