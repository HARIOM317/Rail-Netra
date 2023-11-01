import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:lottie/lottie.dart';
import 'package:railway/src/utils/constants.dart';

class Announcement extends StatefulWidget {
  const Announcement({super.key});

  @override
  State<Announcement> createState() => _AnnouncementState();
}

class _AnnouncementState extends State<Announcement> {
  static String? selectedPlatform;
  TextEditingController announcementController = TextEditingController();
  FlutterTts flutterTts = FlutterTts();

  void textToSpeech(String text) async{
    await flutterTts.setLanguage('en-US');
    await flutterTts.setVolume(0.8);
    await flutterTts.setSpeechRate(0.4);
    await flutterTts.setPitch(1);
    await flutterTts.speak(text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: primaryColor),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: Text(
          "Announcement",
          style: TextStyle(
              color: primaryColor,
              fontWeight: FontWeight.bold,
              fontSize: 22),
        ),
      ),
      body: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Center(
              child: Lottie.asset("assets/animations/announcement.json", animate: true, width: 300),
            ),

            const SizedBox(height: 60,),
            
            Column(
              children: [
                platformDropdownList(selectedPlatform: selectedPlatform, callbackFunction: (String? newPlatform){
                  setState(() {
                    selectedPlatform = newPlatform!;
                  });
                }),

                Padding(
                  padding: const EdgeInsets.only(
                      top: 10, bottom: 10, left: 10, right: 10),

                  child: Card(
                    elevation: 2,
                    color: const Color(0xfffffafa),
                    shadowColor: Colors.grey,
                    shape: RoundedRectangleBorder(
                      side: BorderSide(color: primaryColor, width: 1),
                        borderRadius: BorderRadius.circular(15)
                    ),
                    child: ConstrainedBox(
                      constraints: BoxConstraints(
                        minWidth: MediaQuery.of(context).size.width * 0.80,
                        maxWidth: MediaQuery.of(context).size.width * 0.93,
                        minHeight: MediaQuery.of(context).size.height * 0.05,
                        maxHeight: MediaQuery.of(context).size.height * 0.10,
                      ),

                      child: SingleChildScrollView(
                        scrollDirection: Axis.vertical,

                        child: Padding(
                          padding: const EdgeInsets.all(3.0),
                          child: TextField(
                            enabled: true,
                            keyboardType: TextInputType.multiline,
                            maxLines: null,
                            controller: announcementController,
                            decoration: InputDecoration(
                              fillColor: Colors.white.withOpacity(0.3),
                              filled: true,
                              hintText: "Enter Your Message",
                              border: InputBorder.none,
                            ),
                            style: const TextStyle(
                                fontSize: 16,
                                ),
                            // textAlign: TextAlign.left,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),

            // Announcement button
            Container(
              margin: const EdgeInsets.all(20),
              height: 60,
              width: double.infinity,
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                    backgroundColor: primaryColor,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30))),
                onPressed: selectedPlatform != null ? () {
                  textToSpeech(announcementController.text);
                } : null,
                child: const Text(
                  "Make Announcement",
                  style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 18),
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
