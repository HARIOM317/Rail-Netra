import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:image_picker/image_picker.dart';
import 'package:liquid_pull_to_refresh/liquid_pull_to_refresh.dart';
import 'package:railway/src/components/primary_button.dart';
import 'package:railway/src/utils/constants.dart';
import 'package:uuid/uuid.dart';

class UpdateUserProfile extends StatefulWidget {
  const UpdateUserProfile({super.key});

  @override
  State<UpdateUserProfile> createState() => _UpdateUserProfile();
}

class _UpdateUserProfile extends State<UpdateUserProfile> {
  final key = GlobalKey<FormState>();
  String? id;
  String? profilePic;
  String? downloadUrl;
  bool isSaving = false;
  bool isValidNumber = false;
  bool isImageSelect = false;

  TextEditingController nameController = TextEditingController();
  TextEditingController aboutController = TextEditingController();
  TextEditingController emailController = TextEditingController();
  TextEditingController contactController = TextEditingController();

  // function to get user name from database
  getName() async {
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

  // function to get user about from database
  getAbout() async {
    await FirebaseFirestore.instance
        .collection('users')
        .where('id', isEqualTo: FirebaseAuth.instance.currentUser!.uid)
        .get()
        .then((value) {
      aboutController.text = value.docs.first['about'];
      id = value.docs.first.id;
    });
  }

  // function to get user email from database
  getEmail() async {
    await FirebaseFirestore.instance
        .collection('users')
        .where('id', isEqualTo: FirebaseAuth.instance.currentUser!.uid)
        .get()
        .then((value) {
      emailController.text = value.docs.first['email'];
      id = value.docs.first.id;
    });
  }

  // function to get user contact from database
  getContact() async {
    await FirebaseFirestore.instance
        .collection('users')
        .where('id', isEqualTo: FirebaseAuth.instance.currentUser!.uid)
        .get()
        .then((value) {
      contactController.text = value.docs.first['phone'];
      id = value.docs.first.id;
    });
  }

  // function to reload to profile page on pull down
  Future<void> _handleRefresh() async {
    // reloading takes some time
    getData();
    getName();
    getEmail();
    getContact();
    getAbout();
    return await Future.delayed(const Duration(seconds: 3));
  }

  @override
  void initState() {
    super.initState();
    getName();
    getAbout();
    getEmail();
    getContact();
    getData();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      // Hide keyboard to tap on screen
      onTap: () => FocusScope.of(context).unfocus(),

      child: Scaffold(
        extendBodyBehindAppBar: true,
        appBar: AppBar(
          backgroundColor: Colors.transparent,
          elevation: 0,
          centerTitle: true,
          leading: IconButton(
            icon: Icon(Icons.arrow_back_ios, color: primaryColor),
            onPressed: () => Navigator.of(context).pop(),
          ),
          title: Text(
            "USER PROFILE",
            style: TextStyle(
                color: primaryColor,
                fontWeight: FontWeight.bold,
                fontSize: 22),
          ),
        ),
        body: SafeArea(
          child: isSaving == true
              ? const Center(child: CircularProgressIndicator())
              : LiquidPullToRefresh(
            onRefresh: _handleRefresh,
            color: Colors.transparent,
            backgroundColor: primaryColor,
            height: 300,
            animSpeedFactor: 3,
            showChildOpacityTransition: true,
                child: ListView(
                  children: [
                    Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Center(
                          child: Form(
                            key: key,
                            child: SingleChildScrollView(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                                children: [
                                  Stack(
                                    children: [
                                      Container(
                                        margin: const EdgeInsets.only(top: 10),
                                        decoration: BoxDecoration(
                                            shape: BoxShape.circle,
                                            color: Colors.white,
                                            border: Border.all(
                                                width: 2, color: Colors.white)),
                                        child: profilePic == null
                                            ? CircleAvatar(
                                                radius: 70,
                                                backgroundColor: secondaryColor,
                                                child: const Icon(
                                                  Icons.person,
                                                  size: 120,
                                                  color: Colors.white,
                                                ),
                                              )
                                            : profilePic!.contains('http')
                                                ? CircleAvatar(
                                                    radius: 70,
                                                    backgroundColor: secondaryColor,
                                                    backgroundImage:
                                                        NetworkImage(profilePic!),
                                                  )
                                                : CircleAvatar(
                                                    radius: 70,
                                                    backgroundColor: secondaryColor,
                                                    backgroundImage: FileImage(
                                                        File(profilePic!)),
                                                  ),
                                      ),
                                      Positioned(
                                          bottom: 5,
                                          right: 5,
                                          child: GestureDetector(
                                            onTap: () async {
                                              isImageSelect = true;
                                              final XFile? pickImage =
                                                  await ImagePicker().pickImage(
                                                      source: ImageSource.gallery,
                                                      imageQuality: 50);

                                              if (pickImage != null) {
                                                setState(() {
                                                  profilePic = pickImage.path;
                                                });
                                              }
                                            },
                                            child: Container(
                                              height: 50,
                                              width: 50,
                                              decoration: BoxDecoration(
                                                  shape: BoxShape.circle,
                                                  border: Border.all(
                                                      width: 2,
                                                      color: Colors.white),
                                                  color: Colors.blue.shade800),
                                              child: const Icon(
                                                Icons.camera_alt_rounded,
                                                color: Colors.white,
                                              ),
                                            ),
                                          )),
                                    ],
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.only(
                                        top: 50, bottom: 10, left: 10, right: 10),
                                    child: TextFormField(
                                      enabled: true,
                                      controller: nameController,
                                      validator: (v) {
                                        if (v!.isEmpty) {
                                          return "User Name";
                                        } else if(v.length > 20){
                                          return "Name is too long";
                                        } else {
                                          return null;
                                        }
                                      },
                                      decoration: InputDecoration(
                                        fillColor: Colors.white.withOpacity(0.3),
                                        filled: true,
                                        prefixIcon: const Icon(Icons.person),
                                        suffixIcon: const Icon(Icons.edit),
                                        hintText: nameController.text,
                                        labelText: "Name",
                                        border: InputBorder.none,
                                        focusedBorder: OutlineInputBorder(
                                            borderRadius: BorderRadius.circular(15),
                                            borderSide: BorderSide(
                                                style: BorderStyle.solid,
                                                color: primaryColor,
                                                width: 1.5)),
                                        enabledBorder: OutlineInputBorder(
                                          borderRadius: BorderRadius.circular(15),
                                          borderSide: const BorderSide(
                                            style: BorderStyle.solid,
                                            color: Color(0xFF909A9E),
                                          ),
                                        ),
                                      ),
                                      style: const TextStyle(
                                          fontSize: 16,
                                          ),
                                      textAlign: TextAlign.left,
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.only(
                                        top: 10, bottom: 10, left: 10, right: 10),
                                    child: TextFormField(
                                      enabled: true,
                                      controller: aboutController,
                                      decoration: InputDecoration(
                                        fillColor: Colors.white.withOpacity(0.3),
                                        filled: true,
                                        prefixIcon: const Icon(Icons.info),
                                        suffixIcon: const Icon(Icons.edit),
                                        labelText: "About",
                                        hintText: aboutController.text,
                                        border: InputBorder.none,
                                        focusedBorder: OutlineInputBorder(
                                            borderRadius: BorderRadius.circular(15),
                                            borderSide: BorderSide(
                                                style: BorderStyle.solid,
                                                color: primaryColor,
                                                width: 1.5)),
                                        enabledBorder: OutlineInputBorder(
                                          borderRadius: BorderRadius.circular(15),
                                          borderSide: const BorderSide(
                                            style: BorderStyle.solid,
                                            color: Color(0xFF909A9E),
                                          ),
                                        ),
                                      ),
                                      style: const TextStyle(
                                          fontSize: 16,
                                          ),
                                      textAlign: TextAlign.left,
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.only(
                                        top: 10, bottom: 10, left: 10, right: 10),
                                    child: TextFormField(
                                      enabled: true,
                                      controller: emailController,
                                      validator: (email) {
                                        if (email!.isEmpty ||
                                            email.length < 8 ||
                                            !email.contains("@") ||
                                            email.contains(" ")) {
                                          return "Invalid email address";
                                        } else {
                                          return null;
                                        }
                                      },
                                      decoration: InputDecoration(
                                        fillColor: Colors.white.withOpacity(0.3),
                                        filled: true,
                                        prefixIcon: const Icon(Icons.email),
                                        suffixIcon: const Icon(Icons.edit),
                                        labelText: "Email",
                                        hintText: emailController.text,
                                        border: InputBorder.none,
                                        focusedBorder: OutlineInputBorder(
                                            borderRadius: BorderRadius.circular(15),
                                            borderSide: BorderSide(
                                                style: BorderStyle.solid,
                                                color: primaryColor,
                                                width: 1.5)),
                                        enabledBorder: OutlineInputBorder(
                                          borderRadius: BorderRadius.circular(15),
                                          borderSide: const BorderSide(
                                            style: BorderStyle.solid,
                                            color: Color(0xFF909A9E),
                                          ),
                                        ),
                                      ),
                                      style: const TextStyle(
                                          fontSize: 16,
                                         ),
                                      textAlign: TextAlign.left,
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.only(
                                        top: 10, bottom: 10, left: 10, right: 10),
                                    child: TextFormField(
                                      enabled: true,
                                      controller: contactController,
                                      validator: (v) {
                                        if (v!.length == 10) {
                                          isValidNumber = true;
                                          return null;
                                        } else {
                                          Fluttertoast.showToast(
                                              msg: "Invalid mobile number");
                                          return "Incorrect Mobile Number";
                                        }
                                      },
                                      decoration: InputDecoration(
                                        fillColor: Colors.white.withOpacity(0.3),
                                        filled: true,
                                        prefixIcon: const Icon(Icons.phone),
                                        suffixIcon: const Icon(Icons.edit),
                                        labelText: "Phone",
                                        hintText: contactController.text,
                                        border: InputBorder.none,
                                        focusedBorder: OutlineInputBorder(
                                            borderRadius: BorderRadius.circular(15),
                                            borderSide: BorderSide(
                                                style: BorderStyle.solid,
                                                color: primaryColor,
                                                width: 1.5)),
                                        enabledBorder: OutlineInputBorder(
                                          borderRadius: BorderRadius.circular(15),
                                          borderSide: const BorderSide(
                                            style: BorderStyle.solid,
                                            color: Color(0xFF909A9E),
                                          ),
                                        ),
                                      ),
                                      style: const TextStyle(
                                          fontSize: 16,
                                          ),
                                      textAlign: TextAlign.left,
                                    ),
                                  ),
                                  Container(
                                    margin: const EdgeInsets.only(top: 20),
                                    child: PrimaryButton(
                                        title: "UPDATE PROFILE",
                                        onPressed: () async {
                                          bool? updateProfile = await showDialog(
                                              context: context,
                                              builder: (BuildContext context) {
                                                return AlertDialog(
                                                  elevation: 10,
                                                  shadowColor: primaryColor
                                                      .withOpacity(0.25),
                                                  backgroundColor: Colors.white,
                                                  shape:
                                                      const RoundedRectangleBorder(
                                                    borderRadius: BorderRadius.all(
                                                        Radius.circular(20.0)),
                                                  ),
                                                  title: const Text('Confirm'),
                                                  content: const Text(
                                                      'Are you sure to update your profile?'),
                                                  actions: <Widget>[
                                                    TextButton(
                                                        onPressed: () {
                                                          Navigator.of(context)
                                                              .pop(false);
                                                          Fluttertoast.showToast(
                                                              msg:
                                                                  "Cancel to Updated Profile");
                                                        },
                                                        child: const Text("No")),
                                                    TextButton(
                                                        onPressed: () async {
                                                          Navigator.of(context)
                                                              .pop(true);

                                                          // todo: updating profile
                                                          // update name on firebase
                                                          if (nameController
                                                                  .text.length >= 3 && nameController.text.length < 21) {
                                                            await FirebaseFirestore
                                                                .instance
                                                                .collection('users')
                                                                .doc(id)
                                                                .update({
                                                              'name': nameController
                                                                  .text
                                                            });
                                                          } else if(nameController.text.length > 20){
                                                            Fluttertoast.showToast(msg: "Name is too long!");
                                                          }
                                                          else {
                                                            Fluttertoast.showToast(
                                                                msg:
                                                                    "Name should contain at least 3 characters!");
                                                          }

                                                          // update mobile number if it is valid
                                                          if (contactController
                                                                  .text.length ==
                                                              10) {
                                                            updateMobileNumber();
                                                          }

                                                          if (aboutController
                                                                  .text.length >=
                                                              3) {
                                                            updateAbout();
                                                          } else {
                                                            Fluttertoast.showToast(
                                                                msg:
                                                                    "About should contain at least 3 characters!");
                                                          }

                                                          // update image on firebase
                                                          if (key.currentState!
                                                              .validate()) {
                                                            if (isImageSelect) {
                                                              profilePic == null
                                                                  ? Fluttertoast
                                                                      .showToast(
                                                                          msg:
                                                                              "Please select an profile picture!")
                                                                  : update();
                                                            }
                                                          }

                                                          isImageSelect = false;
                                                          Fluttertoast.showToast(
                                                              msg:
                                                                  "Profile Updated Successfully");
                                                        },
                                                        child: const Text("Yes")),
                                                  ],
                                                );
                                              });

                                          return updateProfile ?? false;
                                        }),
                                  )
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
        ),
      ),
    );
  }

  Future<String?> uploadImage(String filePath) async {
    try {
      final fileName = const Uuid().v4();
      final Reference fbStorage =
          FirebaseStorage.instance.ref('profile').child(fileName);
      final UploadTask uploadTask = fbStorage.putFile(File(filePath));

      await uploadTask.then((p0) async {
        downloadUrl = await fbStorage.getDownloadURL();
      });
      return downloadUrl;
    } catch (e) {
      Fluttertoast.showToast(msg: "Something went wrong!");
    }
    return null;
  }

  update() async {
    setState(() {
      isSaving = true;
    });

    uploadImage(profilePic!).then((value) {
      Map<String, dynamic> data = {
        'name': nameController.text,
        'profilePic': downloadUrl
      };
      FirebaseFirestore.instance
          .collection('users')
          .doc(FirebaseAuth.instance.currentUser!.uid)
          .update(data);
    });

    setState(() {
      isSaving = false;
    });
  }

  updateEmail() async {
    // update phone number on firebase
    await FirebaseFirestore.instance
        .collection('users')
        .doc(id)
        .update({'email': emailController.text});
  }

  updateMobileNumber() async {
    // update phone number on firebase
    await FirebaseFirestore.instance
        .collection('users')
        .doc(id)
        .update({'phone': contactController.text});
  }

  updateAbout() async {
    // update phone number on firebase
    await FirebaseFirestore.instance
        .collection('users')
        .doc(id)
        .update({'about': aboutController.text});
  }
}
