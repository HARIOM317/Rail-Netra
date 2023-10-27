import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/pages/drawer_widgets/terms_and_conditions_page.dart';
import 'package:railway/src/utils/constants.dart';

class PrivacyPolicy extends StatelessWidget{
  const PrivacyPolicy({super.key});


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: primaryColor,
        elevation: 0,
        centerTitle: true,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: bgColor),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: ListView(
        children: [
          Container(
            color: primaryColor,
            height: 300,
            width: double.infinity,
            child: const Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Padding(
                  padding: EdgeInsets.all(10.0),
                  child: CircleAvatar(
                    radius: 40,
                    backgroundImage: AssetImage("assets/images/icons/icon.png"),
                  ),
                ),
                Padding(
                  padding: EdgeInsets.all(10.0),
                  child: Text("Privacy Policy", style: TextStyle(color: Colors.white, fontSize: 30),),
                ),
              ],
            ),
          ),

          const SizedBox(height: 50,),

          writeHeading("Privacy Policy for Rail Netra"),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
            child: Text("Your privacy is important to us.", style: TextStyle(
              color: secondaryColor,
              fontSize: 18
            ),
            ),
          ),

          const SizedBox(height: 20,),

          writeSubHeading("Last Updated: 04-10-23"),

          writeDescription('Rail Netra ("we," "us," or "our") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your personal information when you use our Rail Netra App.'),

          writeDescription("By accessing or using the App, you consent to the practices described in this Privacy Policy"),

          Padding(
            padding: const EdgeInsets.only(left: 16, right: 16, top: 8, bottom: 8),
            child: RichText(
              text: TextSpan(
                // Default Style
                  style: const TextStyle(
                      fontSize: 16,
                      color: Colors.black,
                      fontFamily: "Roboto-Regular"
                  ),

                  children: <TextSpan>[
                    const TextSpan(text: "This Policy shall, at all times be read and construed in consonance and along with the terms of use and access available at ",
                    ),

                    TextSpan(
                        text: "T&Cs",
                        style: TextStyle(
                            color: secondaryColor,
                            decoration: TextDecoration.underline
                        ),
                        recognizer: TapGestureRecognizer()..onTap = () async{
                          goTo(context, const TermsAndConditions());
                        }
                    ),
                    const TextSpan(text: ".")
                  ]
              ),
            ),
          ),

          writeHeading("1. INFORMATION WE COLLECT"),
          writeSubHeading("1.1 PERSONAL INFORMATION:"),
          writeDescription("We may collect personal information from users when they register on the app, subscribe to our services, or interact with certain features of the app. This information may include your name, email address, contact number, and other details necessary for providing you with our services."),
          writeSubHeading("1.2 USER GENERATED CONTENT"),
          writeDescription("We collect user generated content, such as reports, feedback, or incident information."),

          writeHeading("2. INFORMATION USES"),
          writeSubHeading("We may use the information we collect for various purposes, including-"),
          writeDescription("● Providing and maintaining the App's functionality.", align: TextAlign.start),
          writeDescription("● Responding to your requests, comments, or questions.", align: TextAlign.start),
          writeDescription("● Sending you important updates, alerts, and notifications.", align: TextAlign.start),
          writeDescription("● Investigating and responding to incidents or reports.", align: TextAlign.start),

          writeHeading("3. SHARING OF INFORMATION"),
          writeSubHeading("We may share your information with-"),
          writeDescription("● Authorized personnel within our organization.", align: TextAlign.start),
          writeDescription("● Third-party service providers who assist us in operating the App and providing services to you.", align: TextAlign.start),
          writeDescription("● Law enforcement agencies or authorities when required by law or in response to legal requests.", align: TextAlign.start),

          writeHeading("4. DATA SECURITY:"),
          writeDescription("We take appropriate data security measures to protect against unauthorized access, alteration, disclosure, or destruction of your personal information. However, no data transmission over the internet or any wireless network can be guaranteed to be 100% secure. While we strive to protect your personal information, we cannot ensure or warrant the security of any information you transmit to us."),

          writeHeading("5. CHILDREN'S PRIVACY:"),
          writeDescription("Rail Netra is not intended for use by children under the age of 18. We do not knowingly collect personal information from children. If you are a parent or guardian and believe that your child has provided us with personal information, please contact us, and we will take steps to remove such information from our records."),

          writeHeading("6. CHANGE TO THIS PRIVACY POLICY:"),
          writeDescription('We may update this Privacy Policy to reflect changes to our practices or for other operational, legal, or regulatory reasons. We will notify you of any material changes by posting the updated Privacy Policy on the App or through other means.'),

          writeHeading("7. CONTACT INFORMATION:"),
          Padding(
            padding: const EdgeInsets.only(left: 16, right: 16, top: 8, bottom: 8),
            child: RichText(
              text: TextSpan(
                // Default Style
                  style: const TextStyle(
                      fontSize: 16,
                      color: Colors.black,
                      fontFamily: "Roboto-Regular"
                  ),

                  children: <TextSpan>[
                    const TextSpan(text: "If you have any questions or concerns about this Privacy Policy, please contact us at "),

                    TextSpan(
                        text: "myrailway.management@gmail.com",
                        style: TextStyle(
                            color: secondaryColor,
                            decoration: TextDecoration.underline
                        ),
                        recognizer: TapGestureRecognizer()..onTap = () async{
                          openEmail();
                        }
                    ),
                    const TextSpan(text: ".\n\n")
                  ]
              ),
            ),
          ),
        ],
      ),
    );
  }
}