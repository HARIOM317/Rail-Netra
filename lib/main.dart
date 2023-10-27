import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:railway/src/db/notification_services.dart';
import 'package:railway/src/db/share_pref.dart';
import 'package:railway/src/pages/splash_screens/main_splash_screen.dart';
import 'package:railway/src/pages/splash_screens/splash_screen.dart';
import 'package:railway/src/utils/constants.dart';
import 'package:railway/src/utils/custom_material_color.dart';

NotificationServices notificationServices = NotificationServices();

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();

  notificationServices.initializeAwesomeNotification();

  notificationServices.setupFirestoreListener1();
  notificationServices.setupFirestoreListener2();
  notificationServices.setupFirestoreListener3();
  notificationServices.setupFirestoreListener4();
  notificationServices.setupFirestoreListener5();

  await MySharedPreference.init();
  runApp(const MyApp());

}


class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    // stop auto rotation
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
    ]);

    // Set default status bar color
    SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle(statusBarColor: primaryColor));

    return MaterialApp(
      title: 'Rail Netra',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        fontFamily: 'Roboto-Regular',
        primarySwatch: createMaterialColor(primaryColor),
        scaffoldBackgroundColor: bgColor
      ),
      // home: const SplashScreen()

      home: FutureBuilder(
          future: MySharedPreference.getUserType(),
          builder: (BuildContext context, AsyncSnapshot snapshot) {
            if (snapshot.data == "") {
              return const SplashScreen();
            }
            if (snapshot.data == "user") {
              return const MainSplashScreen();
            }
            return progressIndicator(context);
          }),
    );
  }
}
