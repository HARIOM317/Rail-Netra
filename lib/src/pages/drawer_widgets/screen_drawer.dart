import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_advanced_drawer/flutter_advanced_drawer.dart';
import 'package:railway/src/db/notification_services.dart';
import 'package:railway/src/pages/drawer_widgets/help_and_support.dart';
import 'package:railway/src/pages/drawer_widgets/privacy_policy_page.dart';
import 'package:railway/src/pages/drawer_widgets/setting_page.dart';
import 'package:railway/src/pages/home_widgets/announcement.dart';
import 'package:railway/src/pages/home_widgets/dashboard/dashboard.dart';
import 'package:railway/src/pages/home_widgets/homepage.dart';
import 'package:railway/src/pages/drawer_widgets/update_user_profile.dart';
import 'package:railway/src/pages/home_widgets/live_cctv.dart';
import 'package:railway/src/pages/home_widgets/platforms.dart';
import 'package:railway/src/utils/constants.dart';

class DrawerScreen extends StatefulWidget {
  const DrawerScreen({super.key});

  @override
  State<DrawerScreen> createState() => _DrawerScreenState();
}

class _DrawerScreenState extends State<DrawerScreen> {
  final _advancedDrawerController = AdvancedDrawerController();

  final key = GlobalKey<FormState>();
  String? id;
  String? profilePic;
  bool isSaving = false;
  TextEditingController nameController = TextEditingController();


  // for notifications
  NotificationServices notificationServices = NotificationServices();

  // function to get user name from database
  getUserName() async {
    await FirebaseFirestore.instance
        .collection('users')
        .where('id', isEqualTo: FirebaseAuth.instance.currentUser!.uid)
        .get()
        .then((value) {
      nameController.text = value.docs.first['name'];
      id = value.docs.first.id;
    });
  }

  // function to get user image from database
  getData() async {
    await FirebaseFirestore.instance
        .collection('users')
        .where('id', isEqualTo: FirebaseAuth.instance.currentUser!.uid)
        .get()
        .then((value) {
      setState(() {
        nameController.text = value.docs.first['name'];
        id = value.docs.first.id;
        profilePic = value.docs.first['profilePic'];
      });
    });
  }


  @override
  void initState() {
    super.initState();

    setState(() {
      getUserName();
      getData();
    });
  }

  @override
  Widget build(BuildContext context) {
    return AdvancedDrawer(
      backdrop: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(color: primaryColor),
      ),
      controller: _advancedDrawerController,
      animationCurve: Curves.linear,
      animationDuration: const Duration(milliseconds: 250),
      animateChildDecoration: true,
      rtlOpening: false,
      // openScale: 1.0,
      disabledGestures: false,
      childDecoration: const BoxDecoration(
        borderRadius: BorderRadius.all(Radius.circular(30)),
      ),

      drawer: SafeArea(
        child: ListTileTheme(
          textColor: Colors.white,
          iconColor: Colors.white,
          child: ListView(
            children: [
              Stack(
                alignment: Alignment.center,
                children: [
                  GestureDetector(
                    onTap: () {
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => const UpdateUserProfile()));
                      _advancedDrawerController.hideDrawer();
                    },
                    child: Container(
                      margin: const EdgeInsets.only(top: 30),
                      decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: Colors.white,
                          border: Border.all(width: 1, color: Colors.white)),
                      child: profilePic == null
                          ? CircleAvatar(
                              backgroundColor: secondaryColor,
                              radius: 55,
                              child: const Icon(
                                Icons.person,
                                size: 80,
                                color: Colors.white,
                              ),
                            )
                          : profilePic!.contains('http')
                              ? CircleAvatar(
                                  radius: 55,
                                  backgroundImage: NetworkImage(profilePic!),
                                  backgroundColor: secondaryColor,
                                )
                              : CircleAvatar(
                                  radius: 55,
                                  backgroundImage: FileImage(File(profilePic!)),
                                  backgroundColor: secondaryColor,
                                ),
                    ),
                  ),
                ],
              ),

              Container(
                margin: const EdgeInsets.only(top: 10, bottom: 35),
                child: Form(
                  key: key,
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
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.white, fontSize: 22),
                  ),
                ),
              ),

              const Divider(
                thickness: 1,
                color: Colors.white30,
              ),

              // Dashboard button
              ListTile(
                onTap: () {
                  goTo(context, const DashboardScreen());
                  _advancedDrawerController.hideDrawer();
                },
                leading: const Icon(Icons.dashboard_rounded),
                title: const Text('Dashboard'),
              ),

              // Platforms button
              ListTile(
                onTap: () {
                  goTo(context, const ChooseOption());
                  _advancedDrawerController.hideDrawer();
                },
                leading: Image.asset(
                  "assets/images/icons/drawer_icons/platform_drawer.png",
                  height: 24,
                ),
                title: const Text("Platforms"),
              ),

              // Live CCTV button
              ListTile(
                onTap: () {
                  goTo(context, const LiveCCTV());
                  _advancedDrawerController.hideDrawer();
                },
                leading: Image.asset(
                  "assets/images/icons/drawer_icons/live_drawer.png",
                  height: 24,
                ),
                title: const Text("Live CCTV"),
              ),

              // Actions button
              ListTile(
                onTap: () {
                  goTo(context, const Announcement());
                  _advancedDrawerController.hideDrawer();
                },
                leading: Image.asset(
                  "assets/images/icons/drawer_icons/announce_drawer.png",
                  height: 24,
                ),
                title: const Text("Actions"),
              ),

              // Help and Support button
              ListTile(
                onTap: () async {
                  goTo(context, const HelpPage());
                  _advancedDrawerController.hideDrawer();
                },
                leading: const Icon(Icons.headset_mic_rounded),
                title: const Text('Help and Support'),
              ),

              // Setting button
              ListTile(
                onTap: () async {
                  goTo(context, const AppSetting());
                  _advancedDrawerController.hideDrawer();
                },
                leading: const Icon(Icons.settings),
                title: const Text('Setting'),
              ),

              const Divider(
                thickness: 1,
                color: Colors.white30,
              ),

              // Privacy Policy button
              ListTile(
                onTap: () {
                  goTo(context, const PrivacyPolicy());
                  _advancedDrawerController.hideDrawer();
                },
                leading: const Icon(Icons.privacy_tip),
                title: const Text('Privacy Policy'),
              ),
            ],
          ),
        ),
      ),

      child: Scaffold(
        extendBodyBehindAppBar: true,
        appBar: AppBar(
          backgroundColor: Colors.transparent,
          elevation: 0,
          centerTitle: true,
          leading: IconButton(
            onPressed: _handleMenuButtonPressed,
            icon: ValueListenableBuilder<AdvancedDrawerValue>(
              valueListenable: _advancedDrawerController,
              builder: (_, value, __) {
                return AnimatedSwitcher(
                  duration: const Duration(milliseconds: 250),
                  child: Icon(
                    value.visible ? Icons.clear : Icons.menu,
                    color: Colors.black,
                    key: ValueKey<bool>(value.visible),
                  ),
                );
              },
            ),
          ),
        ),
        body: const Homepage(),
      ),
    );
  }

  void _handleMenuButtonPressed() {
    _advancedDrawerController.showDrawer();
  }
}
