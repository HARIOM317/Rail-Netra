class UserModel {
  String? name;
  String? id;
  String? phone;
  String? email;
  String? type;
  String? profilePic;
  String? about;
  String? pushToken;

  UserModel({this.name, this.id, this.phone, this.email, this.type, this.profilePic, this.about, this.pushToken});

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'id': id,
      'phone': phone,
      'email': email,
      'type': type,
      'profilePic': profilePic,
      'about': about,
      'push_token': pushToken
    };
  }

  UserModel.fromJson(Map<String, dynamic> json){
    name = json['name'] ?? '';
    phone = json['phone'] ?? '';
    email = json['email'] ?? '';
    type = json['type'] ?? '';
    profilePic = json['profilePic'] ?? '';
    about = json['about'] ?? '';
    pushToken = json['push_token'] ?? '';
    id = json['id'] ?? '';
  }
}
