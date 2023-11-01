import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:railway/src/db/share_pref.dart';
import 'package:railway/src/user/login_screen.dart';
import 'package:railway/src/utils/custom_page_route.dart';
import 'package:url_launcher/url_launcher.dart';

Color primaryColor = const Color(0xff0076cb);
Color secondaryColor = const Color(0xff03a9f4);
Color bgColor = const Color(0xfffafdfd);

// function to add dropdown list of total platforms
Widget platformDropdownList({required String? selectedPlatform, required Function(String? newPlatform) callbackFunction}){
  return Padding(
    padding: const EdgeInsets.all(15),
    child: DecoratedBox(
      decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(10),
          boxShadow: const <BoxShadow>[
            BoxShadow(color: Color(0xffafafaf), blurRadius: 2)
          ]),
      child: Padding(
        padding: const EdgeInsets.only(left: 30, right: 30),
        child: Center(
          child: DropdownButton<String>(
              borderRadius: BorderRadius.circular(20),
              isExpanded: true,
              hint: const Text("Choose Platform"),
              value: selectedPlatform,
              icon: const Icon(
                Icons.arrow_drop_down,
                size: 35,
              ),
              dropdownColor: Colors.white,
              underline: Container(), //empty line
              style: TextStyle(fontSize: 18, color: primaryColor),
              iconEnabledColor: primaryColor,
              items: const [
                DropdownMenuItem<String>(
                    value: "Platform 1", child: Text("Platform 1")),
                DropdownMenuItem<String>(
                    value: "Platform 2", child: Text("Platform 2")),
                DropdownMenuItem<String>(
                    value: "Platform 3", child: Text("Platform 3")),
                DropdownMenuItem<String>(
                    value: "Platform 4", child: Text("Platform 4")),
                DropdownMenuItem<String>(
                    value: "Platform 5", child: Text("Platform 5")),
                DropdownMenuItem<String>(
                    value: "Platform 6", child: Text("Platform 6")),
              ],
              onChanged: callbackFunction
              ),
        ),
      ),
    ),
  );
}

Widget filteredPlatformDropdownList({required String? selectedPlatform, required Function(String? newPlatform) callbackFunction}){
  return DropdownButton<String>(
      borderRadius: BorderRadius.circular(20),
      hint: Text("Available CCTVs", style: TextStyle(fontSize: 18, color: primaryColor),),
      value: selectedPlatform,
      icon: const Icon(
        Icons.arrow_drop_down,
      ),
      dropdownColor: Colors.white,
      underline: Container(), //empty line
      style: TextStyle(fontSize: 18, color: primaryColor),
      iconEnabledColor: primaryColor,
      items: const [
        DropdownMenuItem<String>(
          value: "Platform 1",
          alignment: Alignment.center,
          child: Text("Platform 1"),
        ),
        DropdownMenuItem<String>(
            value: "Platform 2",
            alignment: Alignment.center,
            child: Text("Platform 2")),
        DropdownMenuItem<String>(
            value: "Platform 3",
            alignment: Alignment.center,
            child: Text("Platform 3")),
        DropdownMenuItem<String>(
            value: "Platform 4",
            alignment: Alignment.center,
            child: Text("Platform 4")),
        DropdownMenuItem<String>(
            value: "Platform 5",
            alignment: Alignment.center,
            child: Text("Platform 5")),
        DropdownMenuItem<String>(
            value: "Platform 6",
            alignment: Alignment.center,
            child: Text("Platform 6")),
      ],
      onChanged: callbackFunction
  );
}

// function to add dropdown list of total monitoring types
Widget monitoringTypeDropdownList({required String? selectedMonitoringType, required Function(String? newPlatform) callbackFunction}){
  return Padding(
    padding: const EdgeInsets.all(15),
    child: DecoratedBox(
      decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(10),
          boxShadow: const <BoxShadow>[
            BoxShadow(color: Color(0xffafafaf), blurRadius: 2)
          ]),
      child: Padding(
        padding: const EdgeInsets.only(left: 30, right: 30),
        child: Center(
          child: DropdownButton<String>(
              borderRadius: BorderRadius.circular(20),
              isExpanded: true,
              hint: const Text("Choose Monitoring Type"),
              value: selectedMonitoringType,
              icon: const Icon(
                Icons.arrow_drop_down,
                size: 35,
              ),
              dropdownColor: Colors.white,
              underline: Container(), //empty line
              style: TextStyle(fontSize: 18, color: primaryColor),
              iconEnabledColor: primaryColor,
              items: const [
                DropdownMenuItem<String>(
                    value: "Crime", child: Text("Crime")),
                DropdownMenuItem<String>(
                    value: "Cleanliness",
                    child: Text("Cleanliness")),
                DropdownMenuItem<String>(
                    value: "Work Monitoring",
                    child: Text("Work Monitoring")),
                DropdownMenuItem<String>(
                    value: "Crowd Management",
                    child: Text("Crowd Management")),
              ],
              onChanged: callbackFunction
          ),
        ),
      ),
    ),
  );
}

// function to show progress indicator
Widget progressIndicator(BuildContext context) {
  return Center(
      child: CircularProgressIndicator(
          backgroundColor: primaryColor.withOpacity(0.5),
          color: primaryColor,
          strokeWidth: 4)
  );
}

// function to show snackbar
void showSnackbar(BuildContext context, String msg, {Color bgColor = Colors.black}) {
  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(
        msg,
        style: const TextStyle(fontSize: 16),
      ),
      backgroundColor: bgColor,
      behavior: SnackBarBehavior.floating));
}

// function to show an alert dialog box
showAlertDialogueBox(BuildContext context, String msg) {
  showDialog(
      context: context,
      builder: (context) => AlertDialog(
            backgroundColor: primaryColor,
            shape: const RoundedRectangleBorder(
              borderRadius: BorderRadius.all(Radius.circular(24)),
            ),
            shadowColor: Colors.deepPurpleAccent.withOpacity(0.5),
            titlePadding: const EdgeInsets.all(25),
            titleTextStyle: const TextStyle(
                color: Colors.white,
                fontSize: 18,
                ),
            title: Text(
              msg,
              textAlign: TextAlign.center,
            ),
          ));
}

//  function to switch page
void goTo(BuildContext context, Widget nextScreen) {
  Navigator.push(context, MaterialPageRoute(builder: (context) => nextScreen));
}

//  function to switch page with left transition
void goToByLeft(BuildContext context, Widget nextScreen) {
  Navigator.push(
      context,
      CustomPageRouteDirection(
          child: nextScreen, direction: AxisDirection.left));
}

Widget introTextDesign1(String text, {double fontSize = 20.0, TextAlign alignment = TextAlign.center}) {
  return Text(
    text,
    style: TextStyle(
        fontSize: fontSize,
        color: primaryColor
    ),
    textAlign: alignment,
  );
}

// function to sign out
signOut(BuildContext context) {
  showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          elevation: 10,
          shadowColor: Colors.black54.withOpacity(0.5),
          backgroundColor: bgColor,
          shape: const RoundedRectangleBorder(
              borderRadius: BorderRadius.all(Radius.circular(15.0))),
          title: const Row(
            children: [
              Padding(
                padding: EdgeInsets.all(8.0),
                child: Icon(
                  Icons.logout,
                  color: Color(0xff6a1010),
                  size: 30,
                ),
              ),
              Padding(
                padding: EdgeInsets.all(8.0),
                child: Text(
                  "Sign out",
                  style: TextStyle(color: Color(0xff6a1010)),
                ),
              )
            ],
          ),
          content: const Text("Do you really want to sign out?"),
          actions: [
            TextButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: Text(
                  "Cancel",
                  style: TextStyle(color: Colors.green[900]),
                )),
            TextButton(
                onPressed: () async {
                  logoutUser();
                  Navigator.pushReplacement(
                      context, CustomPageRoute(child: const LoginScreen()));
                },
                child: const Text(
                  "OK",
                  style: TextStyle(color: Color(0xff6a1010)),
                ))
          ],
        );
      });
}

// function to redirect on gmail for feedback
openEmail() async {
  String email = Uri.encodeComponent("myrailway.management@gmail.com");
  String subject = Uri.encodeComponent("Request for Rail Netra community feedback");
  String body = Uri.encodeComponent("Dear Rail Netra community, \n\n");
  Uri mail = Uri.parse("mailto:$email?subject=$subject&body=$body");
  if (await launchUrl(mail)) {
    //email app opened
  } else {
    Fluttertoast.showToast(msg: "Something went wrong");
  }
}


// function to write heading
Widget writeHeading(String title) {
  return SizedBox(
    width: double.infinity,
    child: Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 20.0),
      child: Text(
        title,
        style: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.bold,
          color: primaryColor,
        ),
        textAlign: TextAlign.start,
      ),
    ),
  );
}

// function to write sub-heading
Widget writeSubHeading(String title) {
  return SizedBox(
    width: double.infinity,
    child: Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
      child: Text(
        title,
        style: const TextStyle(
            fontSize: 18,
            color: Colors.black,
            fontWeight: FontWeight.w500),
        textAlign: TextAlign.start,
      ),
    ),
  );
}

// function to write description
Widget writeDescription(String text, {TextAlign align = TextAlign.justify}) {
  return Container(
    width: double.infinity,
    padding: const EdgeInsets.only(left: 16, right: 16, top: 8, bottom: 8),
    child: Center(
      child: Text(
        text,
        style: const TextStyle(
            fontSize: 16,
            color: Colors.black54),
        textAlign: align,
      ),
    ),
  );
}

