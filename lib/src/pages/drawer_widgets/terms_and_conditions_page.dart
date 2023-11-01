import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/pages/drawer_widgets/privacy_policy_page.dart';
import 'package:railway/src/utils/constants.dart';

class TermsAndConditions extends StatelessWidget{
  const TermsAndConditions({super.key});

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
            child: const Center(
              child: Column(
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
                    child: Text("Terms and Conditions", style: TextStyle(color: Colors.white, fontSize: 30),),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 50,),

          writeHeading("Terms and Conditions for Rail Netra"),
          const SizedBox(height: 20,),

          writeSubHeading("Last Updated: 04-10-23"),

          writeHeading("1. ACCEPTANCE OF TERMS"),
          writeDescription("By accessing or using the Rail Netra app, you agree to comply with and be bound by these Terms and Conditions. If you do not agree with any part of these terms, please do not use the App."),

          writeHeading("2. USE OF THE APP"),
          writeSubHeading("2.1 Eligibility:"),
          writeDescription("You must be at least 18 years of age to use the App. By using the App, you represent and warrant that you are at least 18 years old."),

          writeSubHeading("2.2 User Account:"),
          writeDescription("To use certain features of the App, you may be required to create a user account. You are responsible for maintaining the confidentiality of your account information and for all activities that occur under your account."),

          writeSubHeading("2.3 Prohibited Activities:"),
          writeSubHeading("You agree not to-"),
          writeDescription("● Use the App for any unlawful or fraudulent purpose.", align: TextAlign.start),
          writeDescription("● Upload, post, or transmit any content that is illegal, harmful, threatening, abusive, defamatory, or otherwise objectionable.", align: TextAlign.start),
          writeDescription("● Attempt to gain unauthorized access to the App or its systems.", align: TextAlign.start),
          writeDescription("● Use the App to impersonate any person or entity or misrepresent your affiliation with any person or entity.", align: TextAlign.start),
          writeDescription("● Disrupt or interfere with the security or integrity of the App or its services.", align: TextAlign.start),

          writeHeading("3. USER-GENERATED CONTENT"),
          writeSubHeading("3.1 Ownership:"),
          writeDescription("Any content you upload, post, or transmit to the App, including reports, feedback, or incident information, remains your property. However, by using the App, you grant us a non-exclusive, worldwide, royalty-free license to use, reproduce, modify, adapt, publish, distribute, and display such content for app-related purposes."),

          writeSubHeading("3.2 Content Moderation: "),
          writeDescription("We reserve the right to review, moderate, and remove user-generated content that violates these Terms and Conditions or is otherwise objectionable."),

          writeHeading("4. PRIVACY"),
          Padding(
            padding: const EdgeInsets.only(left: 16, right: 16, top: 8, bottom: 8),
            child: RichText(
              text: TextSpan(
                // Default Style
                  style: const TextStyle(
                      fontSize: 16,
                      color: Colors.black54,
                    fontFamily: "Roboto-Regular"
                  ),

                  children: <TextSpan>[
                    const TextSpan(text: "Your use of the App is governed by our Privacy Policy, which can be found at "),

                    TextSpan(
                        text: "Privacy Policy",
                        style: TextStyle(
                            color: secondaryColor,
                            decoration: TextDecoration.underline
                        ),
                        recognizer: TapGestureRecognizer()..onTap = () async{
                          goTo(context, const PrivacyPolicy());
                        }
                    ),
                    const TextSpan(text: ". "),
                    const TextSpan(text: " Please review the Privacy Policy to understand how we collect, use, and protect your personal information."),
                  ]
              ),
            ),
          ),

          writeHeading("5. INTELLECTUAL PROPERTY"),
          writeDescription("All intellectual property rights in the App, including but not limited to copyrights, trademarks, and patents, are owned by Rail Netra or its licensors. You may not reproduce, distribute, modify, or create derivative works based on the App without our explicit permission."),

          writeHeading("6. DISCLAIMERS AND LIMITATION OF LIABILITY"),
          writeSubHeading("6.1 No Warranty:"),
          writeDescription('The App is provided on an "as-is" and "as-available" basis. We make no warranties, express or implied, regarding the accuracy, reliability, or availability of the App.'),

          writeSubHeading("6.2 Limitation of Liability:"),
          writeDescription("To the extent permitted by law, we shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from or related to the use of the App."),

          writeHeading("7. GOVERNING LAW AND DISPUTE RESOLUTION"),
          writeDescription("These Terms and Conditions are governed by the laws of Indian Government. Any disputes arising out of or relating to these terms shall be resolved through arbitration in accordance with the rules of the Ministry of Railway, and judgment upon the award rendered by the arbitrator(s) may be entered in any court having jurisdiction thereof."),

          writeHeading("8. CHANGES TO TERMS AND CONDITIONS"),
          writeDescription("We reserve the right to modify or update these Terms and Conditions at any time. Any changes will be effective immediately upon posting on the App. It is your responsibility to review these terms periodically."),

          writeHeading("9. CONTACT US"),
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
                    const TextSpan(text: "If you have any questions or concerns about this Terms and Conditions, please contact us at "),

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