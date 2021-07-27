# NTU-X

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ntu60-x-towards-skeleton-based-recognition-of/skeleton-based-action-recognition-on-ntu60-x)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu60-x?p=ntu60-x-towards-skeleton-based-recognition-of)

This repository contains details and pretrained models for the newly introduced dataset NTU-X, which is an extended version of popular NTU dataset. For additional details and results on experiments using NTU-X, take a look at the paper [NTU60-X: Towards Skeleton-based Recognition of Subtle Human Actions](https://arxiv.org/pdf/2101.11529.pdf)

### What is NTU-X

The original NTU dataset contains the human action skeleton which are captured using the Kinect. These skeletons have <b>25 joints</b>. However, all the current top performing models seem to be bottlenecked at certain classes which involve finer finger level movements such as, reading, writing, eat meal etc.

Hence the new NTU-X dataset, introduces a more detailed <b>118 joints</b> skeleton for the action sequences of the NTU dataset. This new dataset, along with 25 body joints, contains <b>42 finger joints</b> and <b>51 face joints</b>.

<img src = "docs/images/NTU-X_sequence_diagram.png"/>

### [Results](https://arxiv.org/pdf/2101.11529.pdf)

<table>
<tr>
  <th>Model</th>
  <th>NTU60</th>
  <th>NTU60-X</th>
  <th>NTU120</th>
  <th>NTU120-X</th>
</tr>
<tr>
  <td>DSTA-Net</td>
  <td>91.50</td>
  <td><b>93.56</b></td>
  <td>86.60</td>
  <td><b>87.80</b></td?>
</tr>
<tr>
  <td>MsG3d</td>
  <td>91.50</td>
  <td><b>91.76</b></td>
  <td>86.90</td>
  <td><b>87.10</b></td?>
</tr>

<tr>
  <td>4s-ShiftGCN</td>
  <td>90.70</td>
  <td><b>91.78</b></td>
  <td>85.90</td>
  <td><b>86.18</b></td>
</tr>

<tr>
  <td>PA-ResGCN</td>
  <td>90.90</td>
  <td><b>91.64</b></td>
  <td><b>87.30</b></td>
  <td>86.42</td>
</tr>


</table>

### Overlayed Examples

<table>
<tr>
<th/>
<td align = "center"><b>NTU</b></th>
<td align = "center"><b>NTU-X (ours)</b></th>
</tr>

<tr>
<td><b>Write</b></td>
<td><img src = "docs/images/write_kinect.gif"/></td>
<td><img src = "docs/images/write_smplx.gif"/></td>
</tr>

<tr>
<td><b>Read</b></td>
<td><img src = "docs/images/read_kinect.gif"/></td>
<td><img src = "docs/images/read_smplx.gif"/></td>
</tr>

<tr>
<td><b>Eat Meal</b></td>
<td><img src = "docs/images/eat_meal_kinect.gif"/></td>
<td><img src = "docs/images/eat_meal_smplx.gif"/></td>
</tr>

</table>


### Comparing NTU-X to other popular action datasets.

<table>

<tr>
<th>Dataset</th>
<th>Body</th>
<th>Fingers</th>
<th>Face</th>
<th># Joints</th>
<th># Sequences</th>
<th># Classes</th>
</tr>

<tr>
<td align = "center">MSR-Action 3d</td>
<td align = "center">:heavy_check_mark:</td>
<td/>
<td/>
<td align = "center">20</td>
<td align = "center">567</td>
<td align = "center">20</td>
</tr>

<tr>
<td align = "center">Northwestern-UCLA</td>
<td align = "center">:heavy_check_mark:</td>
<td/>
<td/>
<td align = "center">24</td>
<td align = "center">1475</td>
<td align = "center">10</td>
</tr>

<tr>
<td align = "center">NTU-RGB+D</td>
<td align = "center">:heavy_check_mark:</td>
<td/>
<td/>
<td align = "center">25</td>
<td align = "center">56880</td>
<td align = "center">60</td>
</tr>

<tr>
<td align = "center">NTU-RGB+D 120</td>
<td align = "center">:heavy_check_mark:</td>
<td/>
<td/>
<td align = "center">25</td>
<td align = "center">114035</td>
<td align = "center">120</td>
</tr>

<tr>
<td align = "center"><b>NTU60-X (Ours)</b></td>
<td align = "center">:heavy_check_mark:</td>
<td align = "center">:heavy_check_mark:</td>
<td align = "center">:heavy_check_mark:</td>
<td align = "center"><b>118</b></td>
<td align = "center"><b>56148</b></td>
<td align = "center"><b>60</b></td>
</tr>

<tr>
<td align = "center"><b>NTU120-X (Ours)</b></td>
<td align = "center">:heavy_check_mark:</td>
<td align = "center">:heavy_check_mark:</td>
<td align = "center">:heavy_check_mark:</td>
<td align = "center"><b>118</b></td>
<td align = "center"><b>113821</b></td>
<td align = "center"><b>120</b></td>
</tr>

</table>

### Download NTU-X dataset

<i>Coming Soon...</i>

### Pretained Models

Few experiments are performed to benchmark this new dataset using the top performing models of the original NTU RGB+D dataset. Details about this models can be found at [Models](./models/)

### The classes included in NTU-X

NTU-X contains same classes as NTU RGB+D dataset. The action labels are mentioned below:

<table>
<tr>
<td align = "center"><b>A1</b> drink water. </td>
<td align = "center"><b>A2</b> eat meal/snack. </td>
<td align = "center"><b>A3</b> brushing teeth. </td>
</tr>

<tr>
<td align = "center"><b>A4</b> brushing hair. </td>
<td align = "center"><b>A5</b> drop. </td>
<td align = "center"><b>A6</b> pickup. </td>
</tr>

<tr>
<td align = "center"><b>A7</b> throw. </td>
<td align = "center"><b>A8</b> sitting down. </td>
<td align = "center"><b>A9</b> standing up (from sitting position).</td>
</tr>

<tr>
<td align = "center"><b>A10</b> clapping. </td>
<td align = "center"><b>A11</b> reading. </td>
<td align = "center"><b>A12</b> writing. </td>
</tr>

<tr>
<td align = "center"><b>A13</b> tear up paper. </td>
<td align = "center"><b>A14</b> wear jacket. </td>
<td align = "center"><b>A15</b> take off jacket. </td>
</tr>

<tr>
<td align = "center"><b>A16</b> wear a shoe. </td>
<td align = "center"><b>A17</b> take off a shoe. </td>
<td align = "center"><b>A18</b> wear on glasses. </td>
</tr>

<tr>
<td align = "center"><b>A19</b> take off glasses. </td>
<td align = "center"><b>A20</b> put on a hat/cap. </td>
<td align = "center"><b>A21</b> take off a hat/cap. </td>
</tr>

<tr>
<td align = "center"><b>A22</b> cheer up. </td>
<td align = "center"><b>A23</b> hand waving. </td>
<td align = "center"><b>A24</b> kicking something. </td>
</tr>

<tr>
<td align = "center"><b>A25</b> reach into pocket. </td>
<td align = "center"><b>A26</b> hopping (one foot jumping). </td>
<td align = "center"><b>A27</b> jump up. </td>
</tr>

<tr>
<td align = "center"><b>A28</b> make a phone call/answer phone. </td>
<td align = "center"><b>A29</b> playing with phone/tablet. </td>
<td align = "center"><b>A30</b> typing on a keyboard. </td>
</tr>

<tr>
<td align = "center"><b>A31</b> pointing to something with finger. </td>
<td align = "center"><b>A32</b> taking a selfie. </td>
<td align = "center"><b>A33</b> check time (from watch). </td>
</tr>

<tr>
<td align = "center"><b>A34</b> rub two hands together. </td>
<td align = "center"><b>A35</b> nod head/bow. </td>
<td align = "center"><b>A36</b> shake head. </td>
</tr>

<tr>
<td align = "center"><b>A37</b> wipe face. </td>
<td align = "center"><b>A38</b> salute. </td>
<td align = "center"><b>A39</b> put the palms together. </td>
</tr>

<tr>
<td align = "center"><b>A40</b> cross hands in front (say stop). </td>
<td align = "center"><b>A41</b> sneeze/cough. </td>
<td align = "center"><b>A42</b> staggering. </td>
</tr>

<tr>
<td align = "center"><b>A43</b> falling. </td>
<td align = "center"><b>A44</b> touch head (headache). </td>
<td align = "center"><b>A45</b> touch chest (stomachache/heart pain).</td>
</tr>

<tr>
<td align = "center"><b>A46</b> touch back (backache). </td>
<td align = "center"><b>A47</b> touch neck (neckache). </td>
<td align = "center"><b>A48</b> nausea or vomiting condition. </td>
</tr>

<tr>
<td align = "center"><b>A49</b> use a fan (with hand or paper)/feeling warm.</td>
<td align = "center"><b>A50</b> punching/slapping other person. </td>
<td align = "center"><b>A51</b> kicking other person. </td>
</tr>

<tr>
<td align = "center"><b>A52</b> pushing other person. </td>
<td align = "center"><b>A53</b> pat on back of other person. </td>
<td align = "center"><b>A54</b> point finger at the other person. </td>
</tr>

<tr>
<td align = "center"><b>A55</b> hugging other person. </td>
<td align = "center"><b>A56</b> giving something to other person. </td>
<td align = "center"><b>A57</b> touch other person's pocket. </td>
</tr>

</tr>
<td align = "center"><b>A58</b> handshaking. </td>
<td align = "center"><b>A59</b> walking towards each other. </td>
<td align = "center"><b>A60</b> walking apart from each other. </td>
</tr>

<tr>
<td align = "center"><b>A61</b> put on headphone.</td>
<td align = "center"><b>A62</b> take off headphone.</td>
<td align = "center"><b>A63</b> shoot at the basket.</td>
</tr>
<tr>
<td align = "center"><b>A64</b> bounce ball.</td>
<td align = "center"><b>A65</b> tennis bat swing.</td>
<td align = "center"><b>A66</b> juggling table tennis balls.</td>
</tr>
<tr>
<td align = "center"><b>A67</b> hush (quite).</td>
<td align = "center"><b>A68</b> flick hair.</td>
<td align = "center"><b>A69</b> thumb up.</td>
</tr>
<tr>
<td align = "center"><b>A70</b> thumb down.</td>
<td align = "center"><b>A71</b> make ok sign.</td>
<td align = "center"><b>A72</b> make victory sign.</td>
</tr>
<tr>
<td align = "center"><b>A73</b> staple book.</td>
<td align = "center"><b>A74</b> counting money.</td>
<td align = "center"><b>A75</b> cutting nails.</td>
</tr>
<tr>
<td align = "center"><b>A76</b> cutting paper (using scissors).</td>
<td align = "center"><b>A77</b> snapping fingers.</td>
<td align = "center"><b>A78</b> open bottle.</td>
</tr>
<tr>
<td align = "center"><b>A79</b> sniff (smell).</td>
<td align = "center"><b>A80</b> squat down.</td>
<td align = "center"><b>A81</b> toss a coin.</td>
</tr>
<tr>
<td align = "center"><b>A82</b> fold paper.</td>
<td align = "center"><b>A83</b> ball up paper</td>
<td align = "center"><b>A84</b> play magic cube.</td>
</tr>
<tr>
<td align = "center"><b>A85</b> apply cream on face.</td>
<td align = "center"><b>A86</b> apply cream on hand back.</td>
<td align = "center"><b>A87</b> put on bag.</td>
</tr>
<tr>
<td align = "center"><b>A88</b> take off bag.</td>
<td align = "center"><b>A89</b> put something into a bag.</td>
<td align = "center"><b>A90</b> take something out of a bag.</td>
</tr>
<tr>
<td align = "center"><b>A91</b> open a box.</td>
<td align = "center"><b>A92</b> move heavy objects.</td>
<td align = "center"><b>A93</b> shake fist.</td>
</tr>
<tr>
<td align = "center"><b>A94</b> throw up cap/hat.</td>
<td align = "center"><b>A95</b> hands up (both hands).</td>
<td align = "center"><b>A96</b> cross arms.</td>
</tr>
<tr>
<td align = "center"><b>A97</b> arm circles.</td>
<td align = "center"><b>A98</b> arm swings.</td>
<td align = "center"><b>A99</b> running on the spot.</td>
</tr>
<tr>
<td align = "center"><b>A100</b>	butt kicks (kick backward).</td>
<td align = "center"><b>A101</b>	cross toe touch.</td>
<td align = "center"><b>A102</b>	side kick.</td>
</tr>
<tr>
<td align = "center"><b>A103</b>	yawn.</td>
<td align = "center"><b>A104</b>	stretch oneself.</td>
<td align = "center"><b>A105</b>	blow nose.</td>
</tr>
<tr>
<td align = "center"><b>A106</b>	hit other person with something.</td>
<td align = "center"><b>A107</b>	wield knife towards other person.</td>
<td align = "center"><b>A108</b>	knock over other person (hit with body).</td>
</tr>
<tr>
<td align = "center"><b>A109</b>	grab other person’s stuff.</td>
<td align = "center"><b>A110</b>	shoot at other person with a gun.</td>
<td align = "center"><b>A111</b>	step on foot.</td>
</tr>
<tr>
<td align = "center"><b>A112</b>	high-five.</td>
<td align = "center"><b>A113</b>	cheers and drink.</td>
<td align = "center"><b>A114</b>	carry something with other person.</td>
</tr>
<tr>
<td align = "center"><b>A115</b>	take a photo of other person.</td>
<td align = "center"><b>A116</b>	follow other person.</td>
<td align = "center"><b>A117</b>	whisper in other person’s ear.</td>
</tr>
<tr>
<td align = "center"><b>A118</b>	exchange things with other person.</td>
<td align = "center"><b>A119</b>	support somebody with hand.</td>
<td align = "center"><b>A120</b> finger-guessing game (playing rock-paper-scissors).</td>
</tr>

</table>


### FAQs

<b>1. How is the NTU-X dataset created?</b>

It is collected by estimating 3D SMPL-X pose outputs from the RGB frames of the NTU-60 RGB videos. We use both[ SMPL-X](https://github.com/vchoutas/smplify-x) and
[Expose](https://github.com/vchoutas/expose) to perform these estimations.

<b>2. How the pose extractor (SMPLx/ExPose) is decided for each class?</b>

We use a semi-automatic approach to estimate the 3D pose for the videos of each class. Keeping the intra-view and intra-subject variance of the NTU dataset in mind, we sample random videos covering each view perclass of NTU and estimate the SMPL-X, ExPose outputs. The estimated skeleton is then backprojected to its
corresponding RGB frame and the accuracy of the alignment is used to select between SMPL-X and Expose.

<b>3. Which class IDs have ExPose used as pose extractor and which class IDs have SMPLx used as pose extractor? </b>

 Empirically, we observe that ExPose,SMPL-X perform equally well for single-person actions but SMPL-X,  though  slow,  provides  better  pose  estimates  for multi-person action class sequences. The classes selected for SMPL-X and Expose are as follows:

 #### Expose

* A1.  drink water
* A2. eat meal/snack
* A3. brushing teeth.
* A4. brushing hair.
* A8. sitting down.
* A9. standing up (from sitting position).
* A12. writing.
* A14. wear jacket.
* A15. take off jacket.
* A16. wear a shoe.
* A17. take off a shoe.
* A18. wear on glasses.
* A19. take off glasses.
* A20. put on a hat/cap.
* A21. take off a hat/cap.
* A24. kicking something.
* A25. reach into pocket.
* A26. hopping (one foot jumping).
* A29. playing with phone/tablet.
* A30. typing on a keyboard.
* A33. check time (from watch).
* A34. rub two hands together.
* A36. shake head.
* A37. wipe face.
* A38. salute.
* A41. sneeze/cough.
* A42. staggering.
* A43. falling.
* A44. touch head (headache).
* A46. touch back (backache).
* A47. touch neck (neckache).
* A48. nausea or vomiting condition.
* A49. use a fan (with hand or paper)/feeling warm.
* A61. put on headphone.
* A62. take off headphone.
* A63. shoot at the basket.
* A64. bounce ball.
* A65. tennis bat swing.
* A68. flick hair.
* A71. make ok sign.
* A72. make victory sign.
* A73. staple book.
* A75. cutting nails.
* A76. cutting paper (using scissors).
* A78. open bottle.
* A79. sniff (smell).
* A80. squat down.
* A82. fold paper.
* A83. ball up paper.
* A84. play magic cube.
* A85. apply cream on face.
* A86. apply cream on hand back.
* A87. put on bag.
* A88. take off bag.
* A89. put something into a bag.
* A90. take something out of a bag.
* A91. open a box.
* A92. move heavy objects.
* A93. shake fist.
* A94. throw up cap/hat.
* A95. hands up (both hands).
* A96. cross arms.
* A97. arm circles.
* A98. arm swings.
* A99. running on the spot.
* A100.	butt kicks (kick backward).
* A101.	cross toe touch.
* A103.	yawn.
* A104.	stretch oneself.
* A105.	blow nose.
* A106.	hit other person with something.
* A107.	wield knife towards other person.
* A108.	knock over other person (hit with body).
* A111.	step on foot.
* A112.	high-five.
* A114.	carry something with other person.
* A119.	support somebody with hand.


#### SMPL-X

 * A5. drop.
 * A6. pickup.
 * A7. throw.
 * A10. clapping.
 * A11. reading.
 * A13. tear up paper.
 * A22. cheer up.
 * A23. hand waving.
 * A27. jump up.
 * A28. make a phone call/answer phone.
 * A31. pointing to something with finger.
 * A32. taking a selfie.
 * A35. nod head/bow.
 * A39. put the palms together.
 * A40. cross hands in front (say stop).
 * A45. touch chest (stomachache/heart pain).
 * A50. punching/slapping other person.
 * A51. kicking other person.
 * A52. pushing other person.
 * A53. pat on back of other person.
 * A54. point finger at the other person.
 * A55. hugging other person.
 * A56. giving something to other person.
 * A57. touch other person's pocket.
 * A58. handshaking.
 * A59. walking towards each other.
 * A60. walking apart from each other.
 * A66. juggling table tennis balls.
 * A67. hush (quite).
 * A69. thumb up.
 * A70. thumb down.
 * A74. counting money.
 * A77. snapping fingers.
 * A81. toss a coin.
 * A102.	side kick.
 * A109.	grab other person’s stuff.
 * A110.	shoot at other person with a gun.
 * A113.	cheers and drink.
 * A115.	take a photo of other person.
 * A116.	follow other person.
 * A117.	whisper in other person’s ear.
 * A118.	exchange things with other person.
 * A120. finger-guessing game (playing rock-paper-scissors). 
