=====================================================================================================

ASVspoof5 Database

Copyright (c) 2024  

ASVspoof5 organizing committee
https://www.asvspoof.org/

=====================================================================================================

1. Directory Structure
_______________________

This package includes
|- flac_T_aa.tar                 ASVspoof 5 training set data
|...
|- flac_T_ae.tar
|
|- flac_D_aa.tar                 ASVspoof 5 development set data
|...
|- flac_D_ac.tar                 
| 
|- flac_E_aa.tar                 ASVspoof 5 evaluation set sdata
|...
|- flac_E_aj.tar      
|
|- ASVspoof5_protocols.tar.gz    ASVspoof 5 protocol and file lists
|
|- LICENSE.txt
|- README.txt


By unpacking flac_T_**.tar, a new directory flac_T will be created, and the training set speech data will be saved in ./flac_T

By unpacking flac_D_**.tar, a new directory flac_D will be created, the development set speech data will be saved in ./flac_D

By unpacking flac_E_**.tar, a new directory flac_E_eval will be created, the evaluation set speech data will be saved in ./flac_E_eval

By unpacking ASVspoof5_protocols.tar.gz, the following files will be extracted
|
|- ASVspoof5.train.tsv           list of training set utterances, keys, and other information
|
|- ASVspoof5.dev.track_1.tsv     list of development set utterances, keys, and other information.
|                                It can be used for model validation in track 1 (detection model without ASV)
|
|- ASVspoof5.dev.trial.txt       list of ASV trials created from development set data.
|                                It can be used for model validation in track 2 (Spoofing robust ASV)
|
|- ASVspoof5.dev.enroll.txt      list of enrollment utterances of targets speaker
|                                in the development set
|
|- ASVspoof5.eval.track_1.tsv    list of evaluation set utterances, keys, and other information.
|                                It is used for track 1 evaluation
|
|- ASVspoof5.eval.trial.tsv      list of ASV trials created from evaluation set data.
|                                It is used for track 2 evaluation
|
|- ASVspoof5.eval.enroll.tsv     list of enrollment utterances of targets speaker
|                                in the evaluation set
|
|- ASVspoof5.codec.config.csv    configurations of codec



2. Statistics and md5sum
_______________________

md5sum                            filename                              number_of_lines
b0cc86b14826a7701b52aad4f53daf9c  ASVspoof5_train/flac_T_aa.tar         36500
d05be3f4be7a343fdbdd0ed29fdff2e1  ASVspoof5_train/flac_T_ab.tar         36500
70d1ba4ad75a20aef3dae541fbd321e3  ASVspoof5_train/flac_T_ac.tar         36500
c9bb56af2cc410d98338d74babcb95c9  ASVspoof5_train/flac_T_ad.tar         36500
a3969816982f3e52300d6147ad796df7  ASVspoof5_train/flac_T_ae.tar         36357
df0be44957623991028cce59792beb17  ASVspoof5_dev/flac_D_aa.tar           47400
1e8cd685d89b64502692f1bcf1a13db3  ASVspoof5_dev/flac_D_ab.tar           47400
5e0031f08c30e4bdbf0c59f91b2d662b  ASVspoof5_dev/flac_D_ac.tar           47334
a8c800766f3d4ef87971e2b4f29663e2  ASVspoof5_eval/flac_E_aa.tar          68188
c35064188f54f07c87aba58de22534a0  ASVspoof5_eval/flac_E_ab.tar          68188
ed7cbcd1b2847998b72472ddf6b445e3  ASVspoof5_eval/flac_E_ac.tar          68188
626a080e994b05df49c577d7b3dede8d  ASVspoof5_eval/flac_E_ad.tar          68188
05f19a5e64fa556714ae1b38eb2ea70b  ASVspoof5_eval/flac_E_ae.tar          68188
2ca2f52a3bbf827f7ec155ecb47e85d6  ASVspoof5_eval/flac_E_af.tar          68188
671200b5de2fc1e74a563b07b057af8f  ASVspoof5_eval/flac_E_ag.tar          68188
eaf885b1299a61eb07f71e82e96dba37  ASVspoof5_eval/flac_E_ah.tar          68188
b2625683bf440abae2af901a45542ad9  ASVspoof5_eval/flac_E_ai.tar          68188
c50281c8233af6f3b0899604d21ade44  ASVspoof5_eval/flac_E_aj.tar          68180

865d0e894ea9f686f0f37e5ae3ae3616  ASVspoof5_protocols.tar.gz
cbfa08e2fab1423e828f102bc97a22f0  ASVspoof5.train.tsv                   182357
e949dee1091a49b04943c548b5cf9860  ASVspoof5.dev.track_1.tsv             140950
11d144e71599f455265bfaa229a3168f  ASVspoof5.dev.track_2.enroll.tsv      398 (1184 utt.)
86b20750e7df2c5607401a98225b8606  ASVspoof5.dev.track_2.trial.tsv       282456
50c0591d45a282cb787a5a4614394f67  ASVspoof5.eval.track_1.tsv            680774
fb754cb8905e7d0816cccaca2d8ffd5f  ASVspoof5.eval.track_2.enroll.tsv     367 (1098 utt.)
de83d6de35d80b58e932fe4ecfccc6e4  ASVspoof5.eval.track_2.trial.tsv      834536

ASVSpoof5 number of files:
total train:
36500 * 4 + 36357 = 182357

total dev:
47400 + 47400 + 47334 = 142134

total eval:
68180 + 68188*9 =681872



Note
a. Speech waveform data are in FLAC format (https://en.wikipedia.org/wiki/FLAC).
   The sampling rate is 16 kHz.

b. In total, 
   flac_T_*.tar: 182357 utterances
   flac_D_*.tar: 142134 utterances 
         = 140950 in ASVspoof5.dev.track_1.tsv + 1184 in ASVspoof5.dev.track_2.enroll.tsv
   flac_E_*.tar: 681872 utterances
         = 680774 in ASVspoof5.eval.track_1.tsv + 1098 in ASVspoof5.eval.track_2.trial.tsv



3. Meta data in txt file
_______________________

ASVspoof5.*.tsv are space-separated

ASVspoof5.train.tsv, ASVspoof5.dev.track_1.tsv and ASVspoof5.eval.track_1.tsv have FIVE columns:

   SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP

   SPEAKER_ID:          T_****, D_****, or E_****        ID of the speaker in the FLAC file
   FLAC_FILE_NAME:      T/D/E_**********                 name of the FLAC file
   SPEAKER_GENDER:      F or M                           gender of the speaker
   CODEC:               C** or -                         name of the codec or compressor condition
   CODEC_Q:             N or -                           codec quality factor configuration number
   CODEC_SEED:          T/D/E_********** or -            if this utterance is coded, name of the original utterance
   ATTACK_TAG:          AC* or -                         tag of the attacker adaptation condition
   ATTACK_LABEL:        A** or bonafide                  name of the attack
   KEY:                 spoof or bonafide                CM key
   TMP:                 -                                reserved column


ASVspoof5.dev.track_2.trial.tsv and ASVspoof5.eval.track_2.trial.tsv have FIVE columns:
    
   TARGET_SPEAKER_ID FLAC_FILE_NAME TARGET_GENDER ATTACK_LABEL ASV_KEY

   TARGET_SPEAKER_ID:   D_****       or E_****           ID of the ASV target speaker
   FLAC_FILE_NAME:      D_********** or E_***********    name of the FLAC file
   TARGET_GENDER:       F or M                           gender of the target speaker
   ATTACK_LABEL         A** or bonafide                  Label of the attack or bonafide
   ASV_KEY:             target, nontarget, or spoof      ASV label



ASVspoof5.dev.enroll.tsv and ASVspoof5.eval.enroll.tsv have TWO columns:

   TARGET_SPEAKER_ID FLAC_FILE_NAMES
   
   TARGET_SPEAKER_ID:   D/E_****                        ID of the ASV target speaker
   FLAC_FILE_NAMES:     D/E_******,D/E_*****,D/E_*****  name of the FLAC files for enrollment

Note
a. FLAC_FILE_NAMES in ASVspoof5.*.enroll.tsv is list of file names separated by ','.
   Some target speakers may have less than 3 enrollment utterances

b. The first column in ASVspoof5.*.track_2.trial.tsv is the ID of the target speaker. 
   The first column in ASVspoof5.*.track_1.tsv and ASVspoof5.train.tsv is the actual speaker ID in the utterance.
   The actual speaker may not be equal to the target speaker when the utterance is used in a 
   non-target AsV trial

c. ASVspoof5.*.track_2.trial.tsv contains limited information on each probe utterance. 
   You may retrieve information (e.g., CODEC) from the corresponding ASVspoof5.*.track_1.tsv

d. for codec quality and the corresponding bit rates, please check ASVspoof5.codec.config.csv


4. Copyright and License
________________________

You are free to use this database under Open Data Commons Attribution License (ODC-By). 

Regarding Open Data Commons Attribution License (ODC-By), please see 
https://opendatacommons.org/licenses/by/1.0/index.html

THIS DATABASE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS DATABASE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The bona fide data is distributed under CC BY 4.0 Deed | Attribution 4.0 International.
Please see https://creativecommons.org/licenses/by/4.0/


5.Ethics
______________________

ASVspoof 5 is committed to upholding ethical standards and to responsible research practices. Our objective is to enhance the security and reliability of automatic speaker verification (ASV) technology by promoting collaboration and progress in the development of robust spoofing/deepfake detection solutions, to promote participation and to protect the interests of stakeholders.

Data contributors and challenge participants are requested to adhere to local data protection regulations when processing speech data. We ask that you conduct your research and development activities in a responsible manner and be mindful of the potential for the misuse of software solutions and results. You are expected to disclose identified vulnerabilities or weaknesses in ASV technology in a responsible manner. The prompt and appropriate reporting of such findings is the shared responsibility of all in our community, contributes to the improvement of security systems and protects against potential misuse.

The ASVspoof organisers explicitly disassociate themselves from any association with or endorsement of hacking activities, unauthorized access attempts, or the creation of spoofs/deepfakes for malicious purposes or personal gain. We strictly condemn any misuse of the knowledge and tools developed through ASVspoof. Any malicious use of challenge outcomes, results or findings is strictly prohibited.

Please note that the ISCA Code of Ethics, available at https://www.isca-speech.org/iscaweb/index.php/about-isca?id=279, applies to all research publications and reports originating from
the ASVspoof initiative and challenge series.


6. Citation
______________________

Please cite the following papers if you use ASVspoof5 database

PLACEHOLDER_FOR_DATABASE_PAPER



Wang, X., Delgado, H., Tak, H., Jung, J.-w., Shim, H.-j., Todisco, M., Kukanov, I., Liu, X., Sahidullah, M., Kinnunen, T.H., Evans, N., Lee, K.A., Yamagishi, J. (2024) ASVspoof 5: crowdsourced speech data, deepfakes, and adversarial attacks at scale. Proc. The Automatic Speaker Verification Spoofing Countermeasures Workshop (ASVspoof 2024), 1-8, doi: 10.21437/ASVspoof.2024-1

@inproceedings{wang24_asvspoof,
  title     = {ASVspoof 5: crowdsourced speech data, deepfakes, and adversarial attacks at scale},
  author    = {Xin Wang and Héctor Delgado and Hemlata Tak and Jee-weon Jung and Hye-jin Shim and Massimiliano Todisco and Ivan Kukanov and Xuechen Liu and Md Sahidullah and Tomi H. Kinnunen and Nicholas Evans and Kong Aik Lee and Junichi Yamagishi},
  year      = {2024},
  booktitle = {The Automatic Speaker Verification Spoofing Countermeasures Workshop (ASVspoof 2024)},
  pages     = {1--8},
  doi       = {10.21437/ASVspoof.2024-1},
}



7. Acknowledgements  
______________________

This database is based on Multilingual LibriSpeech (MLS) [1].

The spoofed data are created by many contributors, and they will be co-authors of a database-description paper.

[1] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “MLS: A large-scale multilingual dataset for speech research,” in Proc. Interspeech, 2020. doi: 10.21437/Interspeech.2020-2826.

