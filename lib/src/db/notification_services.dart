import 'package:flutter/material.dart';
import 'package:awesome_notifications/awesome_notifications.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:intl/intl.dart';
import 'package:railway/src/utils/constants.dart';

class NotificationServices{
  FirebaseFirestore firestore = FirebaseFirestore.instance;
  static bool isCrimeNotificationInitialized = false;
  static bool isWorkMonitoringNotificationInitialized = false;
  static bool isCrowdManagementNotificationInitialized = false;
  static bool isCleanlinessNotificationInitialized = false;
  static bool isCCCTVUrlNotificationInitialized = false;

  int id = 1;

  // String platform = 'Platform';
  String currentTime = DateFormat.jm().format(DateTime.now());
  String currentDate = DateFormat.yMd().format(DateTime.now());
  String createdAt = DateTime.now().millisecondsSinceEpoch.toString();

  void initializeAwesomeNotification(){
    AwesomeNotifications().initialize(null, [
      NotificationChannel(
        channelGroupKey: 'basic_test',
        channelKey: 'high_importance_channel',
        channelName: 'Basic Notifications',
        channelDescription: 'Channel for basic notifications',
        channelShowBadge: true,
        importance: NotificationImportance.High,
        defaultColor: primaryColor,
        ledColor: Colors.white,
        playSound: true,
        onlyAlertOnce: true,
        enableVibration: true,
        enableLights: true,
        defaultPrivacy: NotificationPrivacy.Public,
        defaultRingtoneType: DefaultRingtoneType.Notification
      ),
    ],
      debug: true,
    );
  }

  void setupFirestoreListener1() {
    firestore.collection('crime_videos').snapshots().listen((event) async {
      for (var change in event.docChanges) {
        if (change.type == DocumentChangeType.added) {

          // FirebaseFirestore.instance
          //     .collection('crime_videos')
          //     .orderBy(FieldPath.documentId, descending: true)
          //     .limit(1)
          //     .snapshots()
          //     .listen((QuerySnapshot querySnapshot)  {
          //   if (querySnapshot.docs.isNotEmpty) {
          //     platform = querySnapshot.docs.first['platform'];
          //   } else {
          //     platform = "Platform";
          //   }
          // });

          // Trigger a notification when a new document is added
          if(isCrimeNotificationInitialized){
            sendNotification('New Crime Detected', 'A new crime is detected on Platform');
          }
        }
      }
    });
  }


  void setupFirestoreListener2() {
    firestore.collection('cctv_urls').snapshots().listen((event) {
      for (var change in event.docChanges) {
        if (change.type == DocumentChangeType.added) {
          // Trigger a notification when a new document is added
          if(isCCCTVUrlNotificationInitialized){
            sendNotification('CCTV Added', 'A new cctv camera is added on platform');
          }
        }
      }
    });
  }

  void setupFirestoreListener3() {
    firestore.collection('cleanliness_videos').snapshots().listen((event) {
      for (var change in event.docChanges) {
        if (change.type == DocumentChangeType.added) {
          // Trigger a notification when a new document is added
          if(isCleanlinessNotificationInitialized){
            sendNotification('Trash Found', 'Garbage and Dirt were found on the Platform');
          }
        }
      }
    });
  }

  void setupFirestoreListener4() {
    firestore.collection('work_monitoring_videos').snapshots().listen((event) {
      for (var change in event.docChanges) {
        if (change.type == DocumentChangeType.added) {
          // Trigger a notification when a new document is added
          if(isWorkMonitoringNotificationInitialized){
            sendNotification('Missing Employee', 'The employee was found missing in his office');
          }
        }
      }
    });
  }

  void setupFirestoreListener5() {
    firestore.collection('crowd_management_videos').snapshots().listen((event) {
      for (var change in event.docChanges) {
        if (change.type == DocumentChangeType.added) {
          // Trigger a notification when a new document is added
          if(isCrowdManagementNotificationInitialized){
            sendNotification('Crowd Detected', 'A crowd of people was found on the Platform');
          }
        }
      }
    });
  }

  void sendNotification(String title, String body) async{
    AwesomeNotifications().createNotification(
      content: NotificationContent(
          id: id++,
          channelKey: 'high_importance_channel',
          title: title,
          body: body,
          wakeUpScreen: true,
          category: NotificationCategory.Message
      ),
    );

    createdAt = DateTime.now().millisecondsSinceEpoch.toString();
    currentTime = DateFormat.jm().format(DateTime.now());
    currentDate = DateFormat.yMd().format(DateTime.now());

    await FirebaseFirestore.instance.collection('notifications').add({
      'title': title,
      'body': body,
      'time': currentTime,
      'date': currentDate,
      'created_at': createdAt
    });
  }
}
