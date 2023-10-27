import 'package:flutter/material.dart';
import 'package:flutter_faq/flutter_faq.dart';
import 'package:railway/src/utils/constants.dart';

class HelpPage extends StatelessWidget{
  const HelpPage({super.key});

  Widget frequentlyAskedQuestion(String ques, String ans){
    return Padding(
      padding: const EdgeInsets.all(10.0),
      child: FAQ(
        question: ques,
        answer: ans,
        showDivider: false,
        queDecoration: BoxDecoration(
            borderRadius: const BorderRadius.only(topLeft: Radius.circular(10), topRight: Radius.circular(10)),
            color: const Color(0xffffffff),
            border: Border.all(width: 1, color: Colors.black26)
        ),
        ansDecoration: BoxDecoration(
            borderRadius: const BorderRadius.only(bottomLeft: Radius.circular(10), bottomRight: Radius.circular(10)),
            color: const Color(0xffffffff),
            border: Border.all(width: 1, color: Colors.black12)
        ),
        queStyle: const TextStyle(fontSize: 16),
        ansStyle: const TextStyle(fontSize: 14),
      ),
    );
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
            "FAQs",
            style: TextStyle(
                color: primaryColor,
                fontWeight: FontWeight.bold,
                fontSize: 20)
          ),
        ),

        body: ListView(
          physics: const BouncingScrollPhysics(),
          children: [
            const SizedBox(
              height: 20,
            ),

            frequentlyAskedQuestion("How can I get technical support for the app?", "You can access technical support by visiting the Help and Support section within the app. Here, you can find contact information for our support team and submit your queries or issues."),

            frequentlyAskedQuestion("Can I report bugs or suggest improvements for the app?", "Absolutely! We welcome feedback from our users. You can report bugs or suggest improvements by emailing myrailway.management@gmail.com. Your input helps us enhance the app."),

            frequentlyAskedQuestion("What should I do if I forget my login credentials?", "If you forget your login credentials, you can use the Forgot Password feature on the app's login screen. Follow the instructions provided to reset your password. For further assistance, contact our support team."),

            frequentlyAskedQuestion("What is the purpose of the app?", "The purpose of this app is to enhance railway station safety and efficiency by monitoring and managing crowd congestion, tracking maintenance and construction work progress, and detecting and reporting criminal activities in real-time."),

            frequentlyAskedQuestion("How does the crowd management feature work?", "The crowd management feature utilizes a combination of sensors, cameras, and data analytics to monitor crowd density and flow. It helps railway authorities identify congested areas and take proactive measures to ensure passenger safety."),

            frequentlyAskedQuestion("What technologies are used for real-time crowd monitoring?", "We uses YOLO algorithm to detect object using CCTV and push an alert message notification using Firebase realtime database to the app if it detect crowd of many people on any platform."),

            frequentlyAskedQuestion("How does the app ensure passenger safety?", "Passenger safety is ensured through real-time monitoring and automated alerts. The app can notify authorities and passengers of potential safety hazards or emergencies, enabling rapid responses."),

            frequentlyAskedQuestion("What features are available for work monitoring?", "We monitor the work of any employee using CCTV camera and if employee is absent for more than 10 minutes that we sent alert message to app with employee data."),

            frequentlyAskedQuestion("How does the app detect and report crimes?", "The app detects crimes through video analytics, suspicious behavior recognition, and incident reporting by users. A trained deep learning modal work behind it. When a potential crime is detected, it triggers automatic alerts to app."),

            frequentlyAskedQuestion("What types of crimes does the app address?", "The app is designed to address a wide range of criminal activities, including theft, vandalism, assault, trespassing, and suspicious behavior."),

            frequentlyAskedQuestion("What happens after a crime is detected?", "After a crime is detected, the app sends real-time alerts to nearby law enforcement personnel and relevant authorities for immediate response. Users can also report incidents through the app to initiate investigations."),

            frequentlyAskedQuestion("What user data is collected, and how is it protected?", "User data collected is limited to what is necessary for app functionality, and it is protected following stringent privacy and security protocols. Data is anonymize and encrypted to ensure user privacy."),

            frequentlyAskedQuestion("How often is the app updated?", "The app is regularly updated to enhance performance, security, and user experience. Updates typically occur every few weeks or months."),

            frequentlyAskedQuestion("Is there a customer support system in place?", "Yes, the app has a dedicated customer support system where users can seek assistance, report issues, or ask questions."),

            const SizedBox(
              height: 20,
            )
          ],
        )
    );
  }
}