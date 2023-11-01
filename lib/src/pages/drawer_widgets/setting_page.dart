import 'package:flutter/cupertino.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:railway/src/pages/drawer_widgets/terms_and_conditions_page.dart';
import 'package:railway/src/pages/drawer_widgets/update_user_profile.dart';
import 'package:railway/src/pages/home_widgets/live_cctv.dart';
import 'package:railway/src/user/forgot_password_page.dart';
import 'package:railway/src/utils/constants.dart';
import 'package:share_plus/share_plus.dart';
import 'package:rating_dialog/rating_dialog.dart';
import 'package:store_redirect/store_redirect.dart';

class AppSetting extends StatefulWidget {
  const AppSetting({super.key});

  @override
  State<AppSetting> createState() => _AppSettingState();
}

class _AppSettingState extends State<AppSetting> {
  Widget accountOption(String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            title,
            style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w500,
                color: Colors.grey[600]),
          ),
          const Icon(
            Icons.arrow_forward_ios,
            color: Colors.grey,
          )
        ],
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
            "Setting",
            style: TextStyle(
                color: primaryColor,
                fontWeight: FontWeight.bold,
                fontSize: 22),
          ),
        ),
        body: Container(
          padding: const EdgeInsets.all(10.0),
          child: ListView(
            children: [
              const SizedBox(height: 10),
              Row(
                children: [
                  CircleAvatar(
                    backgroundColor: secondaryColor,
                    child: Icon(
                      Icons.person,
                      color: bgColor,
                      size: 30,
                    ),
                  ),
                  const SizedBox(
                    width: 10,
                  ),
                  const Text(
                    "Account",
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                  ),
                ],
              ),

              const Divider(
                height: 20,
                thickness: 1,
              ),
              const SizedBox(height: 10),

              GestureDetector(
                onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => const UpdateUserProfile()));
                },
                child: accountOption("Update Profile"),
              ),

              GestureDetector(
                  onTap: () {
                    goTo(context, const LiveCCTV());
                  },
                  child: accountOption("Manage Live CCTVs")),

              GestureDetector(
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(builder: (context) => const ForgotPasswordPage()));
                  },
                  child: accountOption("Change Password")),

              const Divider(
                height: 20,
                thickness: 1,
              ),
              const SizedBox(height: 30),

              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  // share button
                  GestureDetector(
                    onTap: () async{
                      await Share.share('com.management.railway');
                    },
                    child: Card(
                      color: Colors.white,
                      elevation: 2,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      child: Container(
                        height: 90,
                        width: 130,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(10),
                          color: primaryColor
                        ),
                        child: const Column(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            Icon(CupertinoIcons.share,
                                color: Colors.white, size: 30),
                            Text(
                              "SHARE",
                              style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold),
                            )
                          ],
                        ),
                      ),
                    ),
                  ),

                  // Rate Us Button
                  GestureDetector(
                    onTap: () {
                      showRatingDialogBox();
                    },
                    child: Card(
                      color: Colors.white,
                      elevation: 2,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10)),
                      child: Container(
                        height: 90,
                        width: 130,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(10),
                          color: primaryColor
                        ),
                        child: const Column(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            Icon(CupertinoIcons.star_fill,
                                color: Colors.white, size: 30),
                            Text(
                              "RATE US",
                              style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold),
                            )
                          ],
                        ),
                      ),
                    ),
                  )
                ],
              ),

              const SizedBox(height: 30),

              // Sign Out Button
              Padding(
                padding: const EdgeInsets.all(10.0),
                child: ElevatedButton(
                  onPressed: () {
                    signOut(context);
                  },
                  style: ElevatedButton.styleFrom(
                      backgroundColor: secondaryColor,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10)),
                      padding: const EdgeInsets.all(15),
                      elevation: 2),
                  child: const Text(
                    "Sign Out",
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                        color: Colors.white),
                  ),
                ),
              ),

              // PrimaryButton(title: "Sign Out", onPressed: (){}),

              const Divider(
                height: 20,
                thickness: 1,
              ),
              const SizedBox(height: 30),

              // Terms and Conditions
              Center(
                child: Padding(
                  padding: const EdgeInsets.only(
                      left: 16, right: 16, top: 8, bottom: 8),
                  child: RichText(
                    text: TextSpan(
                      // Default Style
                        style: const TextStyle(
                            fontSize: 15,
                            color: Color(0xff383838)),
                        children: <TextSpan>[
                          TextSpan(
                              text: "Terms & Conditions",
                              style: TextStyle(
                                  color: secondaryColor,
                                  decoration: TextDecoration.underline),
                              recognizer: TapGestureRecognizer()
                                ..onTap = () async {
                                  goTo(context, const TermsAndConditions());
                                }),
                          const TextSpan(text: " of using the product"),
                        ]),
                  ),
                ),
              ),
            ],
          ),
        ));
  }

  void showRatingDialogBox() {
    showDialog(
        context: context,
        barrierDismissible: true,
        builder: (context) {
          return RatingDialog(
              title: const Text(
                "RATE US",
                textAlign: TextAlign.center,
                style: TextStyle(
                    fontSize: 25,
                    fontWeight: FontWeight.bold),
              ),
              message: const Text(
                "Share your experience",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 20,
                ),
              ),
              image: Image.asset(
                "assets/images/icons/logo.png",
                height: 100,
              ),
              submitButtonText: "SUBMIT",
              submitButtonTextStyle: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold
              ),
              starSize: 35,
              onSubmitted: (response) {
                // print("Rating : ${response.rating}");
                // print("Comment : ${response.comment}");
                if(response.rating < 3.0){

                  showDialog(
                      context: context,
                      builder: (BuildContext context) {
                        return Dialog(
                          shape: RoundedRectangleBorder(
                              borderRadius:
                              BorderRadius.circular(20.0)), //this right here
                          child: Container(
                            decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(20)
                            ),
                            height: 400,
                            child: Padding(
                              padding: const EdgeInsets.all(12.0),
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Padding(
                                    padding: const EdgeInsets.all(8.0),
                                    child: Center(
                                      child: Lottie.asset("assets/animations/improve_app.json", animate: true, width: 200),
                                    ),
                                  ),

                                  const TextField(
                                    decoration: InputDecoration(
                                        border: InputBorder.none,
                                        hintText: 'How can we improve our app?'),
                                  ),
                                  SizedBox(
                                    width: 300.0,
                                    height: 50.0,
                                    child: ElevatedButton(
                                      onPressed: () {
                                        openEmail();
                                        Navigator.pop(context);
                                      },
                                      child: const Text(
                                        "Write your issue",
                                        style: TextStyle(color: Colors.white, fontSize: 18),
                                        textAlign: TextAlign.center,
                                      ),
                                    ),
                                  )
                                ],
                              ),
                            ),
                          ),
                        );
                      });
                } else {
                  StoreRedirect.redirect(androidAppId: "com.management.railway", iOSAppId: "com.management.railway");
                }
              });
        });
  }
}
