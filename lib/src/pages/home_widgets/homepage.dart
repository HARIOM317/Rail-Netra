import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:railway/src/db/notification_services.dart';
import 'package:railway/src/pages/home_widgets/announcement.dart';
import 'package:railway/src/pages/home_widgets/dashboard/dashboard.dart';
import 'package:railway/src/pages/home_widgets/live_cctv.dart';
import 'package:railway/src/pages/home_widgets/notification_page.dart';
import 'package:railway/src/pages/home_widgets/platforms.dart';
import 'package:railway/src/pages/home_widgets/reports.dart';
import 'package:railway/src/utils/constants.dart';
import 'package:railway/src/utils/my_button.dart';

class Homepage extends StatefulWidget {
  const Homepage({super.key});

  @override
  State<Homepage> createState() => _HomepageState();
}

class _HomepageState extends State<Homepage> {
  final key = GlobalKey<FormState>();
  String? id;
  TextEditingController nameController = TextEditingController();

  String currentHour = DateFormat.H().format(DateTime.now());
  String currentMinute = DateFormat.m().format(DateTime.now());

  String greet = "Hello";

  updateTime() async{
    if(int.parse(currentHour) >= 1 && int.parse(currentHour) < 5){
      greet = "Good Night";
    } else if(int.parse(currentHour) >= 5 && int.parse(currentHour) < 12){
      greet = "Good Morning";
    } else if(int.parse(currentHour) >= 12 && int.parse(currentHour) < 17){
      greet = "Good Afternoon";
    } else if(int.parse(currentHour) >= 17 && int.parse(currentHour) < 24){
      greet = "Good Evening";
    } else{
      greet = "Hello";
    }
  }
  // function to get user name from database
  getName() async {
    await FirebaseFirestore.instance
        .collection('users')
        .where('id', isEqualTo: FirebaseAuth.instance.currentUser!.uid)
        .get()
        .then((value) {
      nameController.text = "$greet, ${value.docs.first['name']}";
      id = value.docs.first.id;
    });
  }

  @override
  void initState() {
    setState(() {
      getName();
      updateTime();

      NotificationServices.isCrimeNotificationInitialized = true;
      NotificationServices.isCleanlinessNotificationInitialized = true;
      NotificationServices.isCrowdManagementNotificationInitialized = true;
      NotificationServices.isWorkMonitoringNotificationInitialized = true;
      NotificationServices.isCCCTVUrlNotificationInitialized = true;
    });
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () => _onBackButtonPressed(context),
      child: Scaffold(
        backgroundColor: Colors.white,
        body: ListView(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: TextFormField(
                enabled: false,
                controller: nameController,
                validator: (v) {
                  if (v!.isEmpty) {
                    return "User Name";
                  } else {
                    return null;
                  }
                },
                decoration: InputDecoration(
                  hintText: nameController.text,
                  border: InputBorder.none,
                ),
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w500,
                    color: primaryColor),
                textAlign: TextAlign.start,
              ),
            ),

            Image.asset("assets/images/img/station.jpg"),

            const SizedBox(
              height: 25,
            ),

            // Buttons Row 1
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // Button 1
                GestureDetector(
                    onTap: () {
                      goTo(context, const DashboardScreen());
                    },
                    child: const MyButton(
                        iconImagePath:
                            "assets/images/icons/btn_icons/dashboard.png",
                        buttonText: "Dashboard")),

                // Button 2
                GestureDetector(
                    onTap: () {
                      goTo(context, const LiveCCTV());
                    },
                    child: const MyButton(
                        iconImagePath: "assets/images/icons/btn_icons/cctv.png",
                        buttonText: "Live")),

                // Button 3
                GestureDetector(
                    onTap: () {
                      goTo(context, const ReportsPage());
                    },
                    child: const MyButton(
                        iconImagePath:
                            "assets/images/icons/btn_icons/reports.png",
                        buttonText: "Reports")),
              ],
            ),

            const SizedBox(
              height: 25,
            ),

            // Buttons Row 2
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // Button 4
                GestureDetector(
                    onTap: () {
                      goTo(context, const NotificationPage());
                    },
                    child: const MyButton(
                        iconImagePath:
                            "assets/images/icons/btn_icons/notification.png",
                        buttonText: "Notifications")),

                // Button 5
                GestureDetector(
                    onTap: () {
                      goTo(context, const ChooseOption());
                    },
                    child: const MyButton(
                        iconImagePath:
                            "assets/images/icons/btn_icons/platforms.png",
                        buttonText: "Platforms")),

                // Button 6
                GestureDetector(
                    onTap: () {
                      goTo(context, const Announcement());
                    },
                    child: const MyButton(
                        iconImagePath:
                            "assets/images/icons/btn_icons/announcement.png",
                        buttonText: "Actions")),
              ],
            ),

            const SizedBox(
              height: 25,
            ),
          ],
        ),
      ),
    );
  }

  Future<bool> _onBackButtonPressed(BuildContext context) async {
    bool? exitApp = await showDialog(
        context: context,
        builder: (BuildContext context) {
          return CupertinoAlertDialog(
            title: const Text('Are you sure?'),
            content: const Text('Do you want to close the app?'),
            actions: <Widget>[
              TextButton(
                  onPressed: () {
                    Navigator.of(context).pop(false);
                  },
                  child: const Text("No")),
              TextButton(
                  onPressed: () {
                    Navigator.of(context).pop(true);
                    exit(0);
                  },
                  child: const Text("Yes"))
            ],
          );
        });

    return exitApp ?? false;
  }
}
